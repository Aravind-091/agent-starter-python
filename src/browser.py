"""Playwright engine behind Jarvis's browser tools.

BrowserManager owns ONE Chromium instance (browser -> context -> pages) for the
lifetime of an agent session. The public surface is intentionally coarse: the
model gets ~20 verbs, not 100 Playwright calls.

Safety and robustness rules baked in here:

* Every operation serializes behind an asyncio lock (one page, many callers).
* JavaScript dialogs (alert/confirm/prompt) are NEVER answered automatically:
  the listener records the pending dialog, other page operations fail fast with
  an actionable "call handle_dialog" error, and an operation that still gets
  blocked by the modal loop returns early (its orphaned task result is
  swallowed).
* Navigation times out at 10s, actions at 5s; failures map to ``BrowserError``
  with recovery instructions instead of raw stack traces.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
import weakref
from collections import deque
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Download,
    Frame,
    Locator,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

logger = logging.getLogger("agent")

DEFAULT_NAV_TIMEOUT = 10.0
DEFAULT_ACTION_TIMEOUT = 5.0

_DIALOG_GRACE = 0.6  # seconds to let an op finish once a dialog appears
_POLL_INTERVAL = 0.01
_NETWORK_LOG_SIZE = 200

_CLICK_ACTIONS = frozenset({"click", "double_click", "right_click", "hover"})

_INTERACTIVE_ROLES = (
    "button",
    "link",
    "textbox",
    "searchbox",
    "checkbox",
    "radio",
    "combobox",
    "listbox",
    "menuitem",
    "tab",
    "switch",
    "slider",
    "option",
    "heading",
)

_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+(:\d+)?([/?#].*)?$")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")

_INSPECT_JS = r"""() => {
  const out = [];
  const els = document.querySelectorAll(
    'a[href], button, input, select, textarea, summary, [role], [onclick], [tabindex]:not([tabindex="-1"])'
  );
  const nameOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria.trim();
    if (el.labels && el.labels.length) return (el.labels[0].innerText || '').trim();
    const tag = el.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') {
      const ph = el.getAttribute('placeholder');
      if (ph) return ph.trim();
    }
    if (tag === 'IMG') {
      const alt = el.getAttribute('alt');
      if (alt) return alt.trim();
    }
    const title = el.getAttribute('title');
    if (title) return title.trim();
    return (el.innerText || el.textContent || '').trim();
  };
  const roleOf = (el) => {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName;
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'A' && el.hasAttribute('href')) return 'link';
    if (tag === 'BUTTON') return 'button';
    if (tag === 'SELECT') return 'combobox';
    if (tag === 'TEXTAREA') return 'textbox';
    if (tag === 'IMG') return 'img';
    if (tag === 'INPUT') {
      if (type === 'button' || type === 'submit' || type === 'reset') return 'button';
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (type === 'range') return 'slider';
      if (type === 'search') return 'searchbox';
      return 'textbox';
    }
    if (/^H[1-6]$/.test(tag)) return 'heading';
    return tag.toLowerCase();
  };
  for (const el of els) {
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) continue;
    const name = nameOf(el).replace(/\s+/g, ' ').slice(0, 120);
    if (!name) continue;
    out.push({ role: roleOf(el), name: name });
    if (out.length >= 40) break;
  }
  return out;
}"""

_STORAGE_LIST_JS = r"""() => {
  const out = {};
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i);
    out[k] = localStorage.getItem(k);
  }
  return out;
}"""

_STORAGE_CLEAR_JS = (
    r"() => { localStorage.clear(); sessionStorage.clear(); return true; }"
)

_BODY_TEXT_JS = r"() => (document.body ? document.body.innerText : '')"

_NAMED_KEYS: dict[str, str] = {
    "enter": "Enter",
    "escape": "Escape",
    "esc": "Escape",
    "tab": "Tab",
    "backspace": "Backspace",
    "delete": "Delete",
    "arrowup": "ArrowUp",
    "arrowdown": "ArrowDown",
    "arrowleft": "ArrowLeft",
    "arrowright": "ArrowRight",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "home": "Home",
    "end": "End",
    "space": " ",
    "f5": "F5",
    "insert": "Insert",
}

_MODIFIERS: dict[str, str] = {
    "ctrl": "Control",
    "control": "Control",
    "shift": "Shift",
    "alt": "Alt",
    "meta": "Meta",
    "cmd": "Meta",
    "command": "Meta",
}


class BrowserError(RuntimeError):
    """A browser failure with an actionable message for the model."""


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n[... truncated]"


def _first(error: Exception) -> str:
    """First line of an error message (Playwright errors carry long call logs)."""
    lines = str(error).splitlines()
    head = lines[0] if lines else error.__class__.__name__
    return _clip(head, 300)


def _swallow_task(task: asyncio.Future[Any]) -> None:
    """Retrieve an abandoned task's exception so nothing is logged as unhandled."""
    if task.cancelled():
        return
    with contextlib.suppress(Exception):
        task.exception()


def validate_http_url(url: str) -> str:
    """Normalize a spoken URL to http(s) or raise BrowserError.

    Accepts bare domains (``example.com`` -> ``https://example.com``) and
    host:port forms; rejects every other scheme with a message that names
    http/https so the model can self-correct.
    """
    raw = (url or "").strip()
    if not raw:
        raise BrowserError(
            "No URL given. Provide an http(s) address, e.g. https://example.com."
        )
    # Reject non-http schemes, but treat "localhost:8080" as host:port, not a scheme.
    if _SCHEME_RE.match(raw) and not re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:\d", raw):
        scheme = raw.split(":", 1)[0].lower()
        if scheme in ("http", "https"):
            return raw
        raise BrowserError(
            f"Only http and https URLs are allowed (got '{scheme}://...'). "
            "Give me the site's normal web address, e.g. https://example.com."
        )
    if _HOST_RE.match(raw):
        return f"https://{raw}"
    raise BrowserError(
        f"'{raw}' is not a valid web address. Only http and https URLs are allowed "
        "- try something like https://example.com."
    )


def headless_from_env(default: bool = True) -> bool:
    """Read ``BROWSER_HEADLESS``; set it to ``false`` to watch Jarvis browse."""
    raw = os.getenv("BROWSER_HEADLESS")
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _normalize_key(key: str) -> str:
    """Map a spoken key name to a Playwright key, or raise BrowserError."""
    raw = (key or "").strip()
    if not raw:
        raise BrowserError(
            "No key given. Examples: Tab, Enter, Escape, ArrowDown, Ctrl+L."
        )
    parts = [p.strip() for p in raw.split("+")] if "+" in raw else [raw]
    *mods, final = parts
    out_mods: list[str] = []
    for mod in mods:
        canonical = _MODIFIERS.get(mod.casefold())
        if canonical is None:
            raise BrowserError(
                f"Key '{key}' is not supported. Modifiers must be Ctrl, Shift, Alt, or Meta."
            )
        out_mods.append(canonical)
    low = final.casefold()
    if low in _NAMED_KEYS:
        final_key = _NAMED_KEYS[low]
    elif len(final) == 1 and final.isalnum():
        final_key = final
    else:
        raise BrowserError(
            f"Key '{key}' is not supported. Use Tab, Enter, Escape, Arrow keys, "
            "PageUp, PageDown, Home, End, Backspace, Delete, Space, F5, or a single "
            "letter/digit (optionally with Ctrl/Shift/Alt/Meta)."
        )
    return "+".join([*out_mods, final_key])


class BrowserManager:
    """Serializes all browser access for one agent session."""

    def __init__(self, *, headless: bool = True) -> None:
        self._headless = headless
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._ctx: BrowserContext | None = None
        self._page: Page | None = None
        self._lock = asyncio.Lock()
        self._pending: Any = None  # playwright Dialog awaiting handle_dialog
        self._dialog_arrived = asyncio.Event()
        self._network: deque[Any] = deque(maxlen=_NETWORK_LOG_SIZE)
        self._downloads: deque[Download] = deque(maxlen=10)
        self._download_arrived = asyncio.Event()
        self._wired: weakref.WeakSet[Any] = weakref.WeakSet()

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Launch Chromium eagerly (session warmup). No-op if already running."""
        async with self._lock:
            await self._ensure()

    async def close(self) -> None:
        """Stop Chromium. Safe to call repeatedly, including before start()."""
        async with self._lock:
            await self._teardown()

    async def _ensure(self) -> None:
        """Start the browser if needed. Caller must hold the lock."""
        if self._pw is not None:
            return
        try:
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(
                headless=self._headless,
                chromium_sandbox=False,
            )
            self._ctx = await self._browser.new_context(accept_downloads=True)
            self._ctx.on("page", self._on_page)
            self._ctx.on("response", self._on_response)
            self._ctx.on("download", self._on_download)
            page = await self._ctx.new_page()
            self._wire(page)
            self._page = page
        except Exception as error:
            await self._teardown()
            raise BrowserError(
                f"Could not start the browser: {_first(error)} "
                "Run `uv run playwright install chromium` if Chromium is missing."
            ) from error

    async def _teardown(self) -> None:
        pw, self._pw = self._pw, None
        self._browser = None
        self._ctx = None
        self._page = None
        self._pending = None
        self._dialog_arrived.clear()
        self._download_arrived.clear()
        self._network.clear()
        self._downloads.clear()
        self._wired.clear()
        if pw is not None:
            with contextlib.suppress(Exception):
                await pw.stop()

    # -- page wiring and event listeners -------------------------------------

    def _wire(self, page: Page) -> None:
        if page in self._wired:
            return
        self._wired.add(page)
        # Never answer dialogs automatically: the model decides via handle_dialog.
        page.on("dialog", self._on_dialog)
        page.set_default_timeout(int(DEFAULT_ACTION_TIMEOUT * 1000))
        page.set_default_navigation_timeout(int(DEFAULT_NAV_TIMEOUT * 1000))

    def _on_page(self, page: Page) -> None:
        self._wire(page)

    def _on_dialog(self, dialog: Any) -> None:
        self._pending = dialog
        self._dialog_arrived.set()

    def _on_response(self, response: Any) -> None:
        try:
            self._network.append(
                {
                    "method": response.request.method,
                    "url": response.url,
                    "status": response.status,
                }
            )
        except Exception:
            logger.debug("dropped a network log entry", exc_info=True)

    def _on_download(self, download: Download) -> None:
        self._downloads.append(download)
        self._download_arrived.set()

    # -- properties ----------------------------------------------------------

    @property
    def active_page(self) -> Page:
        page = self._page
        if page is None:
            raise BrowserError("The browser has not started yet - call open_url first.")
        return page

    @property
    def pending_dialog(self) -> dict[str, Any] | None:
        """The open alert/confirm/prompt, if any (never auto-answered)."""
        dialog = self._pending
        if dialog is None:
            return None
        return {"type": dialog.type, "message": dialog.message}

    # -- orchestration -------------------------------------------------------

    def _require_dialog_free(self, what: str) -> None:
        dialog = self._pending
        if dialog is not None:
            raise BrowserError(
                f"A browser dialog is open: '{dialog.message}' - {what} cannot run. "
                "Call handle_dialog(action='accept'|'dismiss') first, then retry."
            )

    async def _run(
        self,
        what: str,
        make_op: Callable[[], Coroutine[Any, Any, Any]],
        *,
        timeout: float,
        nominal: dict[str, Any] | None = None,
        check_dialog: bool = True,
    ) -> Any:
        """Run one serialized browser operation with dialog-aware polling."""
        async with self._lock:
            await self._ensure()
            if check_dialog:
                self._require_dialog_free(what)
            return await self._dispatch(
                make_op(), what=what, timeout=timeout, nominal=nominal
            )

    async def _dispatch(
        self,
        op: Coroutine[Any, Any, Any],
        *,
        what: str,
        timeout: float,
        nominal: dict[str, Any] | None = None,
    ) -> Any:
        task = asyncio.ensure_future(op)
        task.add_done_callback(_swallow_task)
        deadline = time.monotonic() + timeout
        blocked_since: float | None = None
        while not task.done():
            now = time.monotonic()
            if self._pending is not None and blocked_since is None:
                blocked_since = now
            if blocked_since is not None and now - blocked_since >= _DIALOG_GRACE:
                # The page is frozen in a modal dialog; stop waiting on this op.
                task.cancel()
                if nominal is not None:
                    return nominal
                message = getattr(self._pending, "message", "?")
                raise BrowserError(
                    f"{what} is blocked by an open browser dialog: '{message}'. "
                    "Call handle_dialog(action='accept'|'dismiss') first, then retry."
                )
            if now >= deadline:
                task.cancel()
                raise BrowserError(
                    f"{what} did not finish within {timeout:g}s - the page may be "
                    "blocked by a dialog or still loading. Try handle_dialog, "
                    "wait_for, or navigate(action='reload')."
                )
            await asyncio.wait({task}, timeout=_POLL_INTERVAL)
        try:
            result = task.result()
        except PlaywrightTimeoutError as error:
            raise BrowserError(
                f"{what} timed out: {_first(error)} If a dialog is open, call "
                "handle_dialog first; otherwise inspect_page and retry."
            ) from error
        except PlaywrightError as error:
            raise BrowserError(f"{what} failed: {_first(error)}") from error
        if result is None and nominal is not None:
            return nominal
        return result

    # -- target resolution ---------------------------------------------------

    @staticmethod
    def _strategies(name: str) -> list[Callable[[Frame], Locator]]:
        if name[0] in "#.[":  # explicit CSS selector passthrough
            return [lambda frame: frame.locator(name)]
        strategies: list[Callable[[Frame], Locator]] = [
            lambda frame: frame.get_by_label(name, exact=False),
            lambda frame: frame.get_by_placeholder(name, exact=False),
            lambda frame: frame.get_by_title(name, exact=False),
            lambda frame: frame.get_by_alt_text(name, exact=False),
        ]
        strategies.extend(
            lambda frame, role=role: frame.get_by_role(role, name=name)
            for role in _INTERACTIVE_ROLES
        )
        if _ID_RE.match(name):
            strategies.append(lambda frame: frame.locator(f"[id='{name}']"))
        strategies.append(lambda frame: frame.get_by_text(name, exact=True))
        strategies.append(lambda frame: frame.get_by_text(name, exact=False))
        return strategies

    async def _resolve(self, target: str) -> Locator:
        """Find an element by accessible name across all frames, or raise."""
        name = (target or "").strip()
        if not name:
            raise BrowserError(
                "Element name is empty. Use inspect_page to list interactable "
                "elements first."
            )
        frames = [frame for frame in self.active_page.frames if not frame.is_detached()]
        for make in self._strategies(name):
            for frame in frames:
                try:
                    locator = make(frame)
                    count = await locator.count()
                except PlaywrightError:
                    continue
                if count:
                    return locator.first
        raise BrowserError(
            f"Could not find an element named '{name}'. Use inspect_page and pass "
            "an exact name from that list."
        )

    # -- navigation and reading ----------------------------------------------

    async def open_url(self, url: str) -> dict[str, Any]:
        target = validate_http_url(url)
        nav_ms = int(DEFAULT_NAV_TIMEOUT * 1000)

        async def op() -> dict[str, Any]:
            page = self.active_page
            try:
                await page.goto(target, wait_until="load", timeout=nav_ms)
                title = await page.title()
            except PlaywrightTimeoutError:
                raise BrowserError(
                    f"The page did not load within {DEFAULT_NAV_TIMEOUT:g}s: "
                    f"{target}. The site may be slow or unreachable - try again, or "
                    "use search_the_web to find an alternative."
                ) from None
            except PlaywrightError as error:
                raise BrowserError(
                    f"Could not open {target}: {_first(error)} Check the address, "
                    "or use search_the_web instead."
                ) from error
            return {"url": page.url, "title": title}

        return await self._run(f"Opening {target}", op, timeout=DEFAULT_NAV_TIMEOUT + 3)

    async def navigate(self, action: str) -> dict[str, Any]:
        verb = (action or "").strip().casefold()
        if verb not in ("back", "forward", "reload"):
            raise BrowserError("action must be 'back', 'forward', or 'reload'.")
        nav_ms = int(DEFAULT_NAV_TIMEOUT * 1000)

        async def op() -> dict[str, Any]:
            page = self.active_page
            try:
                if verb == "back":
                    await page.go_back(wait_until="load", timeout=nav_ms)
                elif verb == "forward":
                    await page.go_forward(wait_until="load", timeout=nav_ms)
                else:
                    await page.reload(wait_until="load", timeout=nav_ms)
            except PlaywrightTimeoutError:
                raise BrowserError(
                    f"Navigation ('{verb}') did not complete within "
                    f"{DEFAULT_NAV_TIMEOUT:g}s. Try again, or wait_for('load')."
                ) from None
            except PlaywrightError as error:
                raise BrowserError(
                    f"Navigation ('{verb}') failed: {_first(error)} There may be no "
                    "history for that direction."
                ) from error
            return {"action": verb, "url": page.url}

        return await self._run(
            f"Navigating ({verb})", op, timeout=DEFAULT_NAV_TIMEOUT + 3
        )

    async def read_page(self) -> dict[str, Any]:
        async def op() -> dict[str, Any]:
            page = self.active_page
            text = await page.evaluate(_BODY_TEXT_JS)
            return {
                "text": _clip(str(text), 6000),
                "title": await page.title(),
                "url": page.url,
            }

        return await self._run(
            "Reading the page", op, timeout=DEFAULT_ACTION_TIMEOUT + 3
        )

    async def inspect_page(self) -> dict[str, Any]:
        async def op() -> dict[str, Any]:
            page = self.active_page
            elements: list[Any] = []
            for frame in page.frames:
                if frame.is_detached():
                    continue
                try:
                    found = await frame.evaluate(_INSPECT_JS)
                except PlaywrightError:
                    continue  # frame navigated away or went cross-origin mid-scan
                if isinstance(found, list):
                    elements.extend(found)
                if len(elements) >= 40:
                    break
            text = await page.evaluate(_BODY_TEXT_JS)
            return {
                "elements": elements[:40],
                "text": _clip(str(text), 400),
                "title": await page.title(),
                "url": page.url,
            }

        return await self._run(
            "Inspecting the page", op, timeout=DEFAULT_ACTION_TIMEOUT + 3
        )

    async def take_screenshot(
        self, full_page: bool = False, target: str | None = None
    ) -> bytes:
        async def op() -> bytes:
            if target:
                locator = await self._resolve(target)
                return await locator.screenshot(type="png")
            return await self.active_page.screenshot(
                full_page=bool(full_page), type="png"
            )

        png = await self._run(
            "Taking a screenshot", op, timeout=DEFAULT_ACTION_TIMEOUT + 3
        )
        if not isinstance(png, bytes):
            raise BrowserError("The screenshot returned no image data - try again.")
        return png

    # -- interacting ----------------------------------------------------------

    async def click(
        self, target: str, *, action: str = "click", force: bool = False
    ) -> dict[str, Any]:
        act = (action or "click").strip().casefold()
        if act not in _CLICK_ACTIONS:
            raise BrowserError(
                f"Unknown click action '{action}'. Use: click, double_click, "
                "right_click, hover."
            )
        nominal: dict[str, Any] = (
            {"hovered": target} if act == "hover" else {"clicked": target}
        )
        verb = {
            "click": "Clicking",
            "double_click": "Double-clicking",
            "right_click": "Right-clicking",
            "hover": "Hovering",
        }[act]

        async def op() -> None:
            locator = await self._resolve(target)
            if act == "hover":
                await locator.hover(force=force)
            elif act == "double_click":
                await locator.dblclick(force=force)
            elif act == "right_click":
                await locator.click(button="right", force=force)
            else:
                await locator.click(force=force)

        return await self._run(
            f"{verb} '{target}'",
            op,
            timeout=DEFAULT_ACTION_TIMEOUT + 3,
            nominal=nominal,
        )

    async def type_text(
        self, target: str, text: str, *, submit: bool = False
    ) -> dict[str, Any]:
        async def op() -> None:
            locator = await self._resolve(target)
            await locator.fill(text)
            if submit:
                await locator.press("Enter")

        return await self._run(
            f"Typing into '{target}'",
            op,
            timeout=DEFAULT_ACTION_TIMEOUT + 3,
            nominal={"typed": target, "submitted": submit},
        )

    async def select_option(self, target: str, option: str) -> dict[str, Any]:
        async def op() -> dict[str, Any]:
            locator = await self._resolve(target)
            tag = await locator.evaluate("el => el.tagName")
            if not isinstance(tag, str) or tag.upper() != "SELECT":
                found = tag.lower() if isinstance(tag, str) else "element"
                raise BrowserError(
                    f"'{target}' is not a standard dropdown (found <{found}>). For "
                    "custom menus, click it to open the menu, then click the option."
                )
            options = await locator.evaluate(
                "el => Array.from(el.options).map(o => "
                "({label: (o.textContent || '').trim(), value: o.value}))"
            )
            want = (option or "").strip().casefold()
            match = next(
                (
                    entry
                    for entry in options
                    if entry.get("label", "").casefold() == want
                    or entry.get("value", "") == option
                ),
                None,
            )
            if match is None:
                available = (
                    ", ".join(entry.get("label", "") for entry in options[:12])
                    or "(no options)"
                )
                raise BrowserError(
                    f"'{option}' is not an option of '{target}'. Available: {available}."
                )
            await locator.select_option(value=match["value"])
            return {"selected": option, "target": target, "value": match["value"]}

        return await self._run(
            f"Selecting '{option}' in '{target}'",
            op,
            timeout=DEFAULT_ACTION_TIMEOUT + 3,
        )

    async def press_key(self, key: str) -> dict[str, Any]:
        canonical = _normalize_key(key)

        async def op() -> None:
            await self.active_page.keyboard.press(canonical)

        return await self._run(
            f"Pressing {canonical!r}",
            op,
            timeout=DEFAULT_ACTION_TIMEOUT + 3,
            nominal={"key": canonical},
        )

    async def scroll(
        self, direction: str, amount: int | None = None, *, target: str | None = None
    ) -> dict[str, Any]:
        if target:

            async def op_to_target() -> None:
                locator = await self._resolve(target)
                await locator.scroll_into_view_if_needed(
                    timeout=int(DEFAULT_ACTION_TIMEOUT * 1000)
                )

            return await self._run(
                f"Scrolling to '{target}'",
                op_to_target,
                timeout=DEFAULT_ACTION_TIMEOUT + 3,
                nominal={"scrolled": "target", "target": target},
            )
        side = (direction or "").strip().casefold()
        if side not in ("up", "down"):
            raise BrowserError(
                "direction must be 'up' or 'down' (or pass a target to scroll an "
                "element into view)."
            )
        try:
            pixels = int(amount) if amount is not None else 600
        except (TypeError, ValueError):
            raise BrowserError(
                f"amount must be a number of pixels, got {amount!r}."
            ) from None
        if pixels <= 0:
            raise BrowserError("amount must be a positive number of pixels.")
        delta = pixels if side == "down" else -pixels

        async def op_by_pixels() -> dict[str, Any]:
            y = await self.active_page.evaluate(
                "(dy) => { window.scrollBy(0, dy); return window.scrollY; }", delta
            )
            return {"direction": side, "amount": pixels, "y": int(y)}

        return await self._run(
            f"Scrolling {side}", op_by_pixels, timeout=DEFAULT_ACTION_TIMEOUT + 3
        )

    async def upload_file(
        self, target: str, path: str, force: bool = False
    ) -> dict[str, Any]:
        del force  # set_input_files needs no actionability workaround
        file_path = Path(str(path)).expanduser()
        if not file_path.is_file():
            raise BrowserError(
                f"File not found: {path}. Provide the full path to an existing file."
            )

        async def op() -> None:
            locator = await self._resolve(target)
            try:
                await locator.set_input_files(str(file_path))
            except PlaywrightError as error:
                raise BrowserError(
                    f"Could not attach the file to '{target}': {_first(error)} "
                    "The target must be a file input - use inspect_page to find it."
                ) from error

        return await self._run(
            f"Uploading {file_path.name} to '{target}'",
            op,
            timeout=DEFAULT_ACTION_TIMEOUT + 3,
            nominal={"uploaded": str(file_path), "target": target},
        )

    # -- waiting ---------------------------------------------------------------

    async def wait_for(
        self, condition: str, value: str | None = None, *, timeout: float | None = None
    ) -> dict[str, Any]:
        wait_s = float(timeout) if timeout is not None else 10.0
        if wait_s <= 0:
            raise BrowserError("timeout must be a positive number of seconds.")
        ms = int(wait_s * 1000)
        cond = (condition or "").strip().casefold()

        if cond == "text":
            if not value:
                raise BrowserError("wait_for('text') needs the text to wait for.")

            async def op_text() -> dict[str, Any]:
                locator = self.active_page.get_by_text(value, exact=False).first
                try:
                    await locator.wait_for(timeout=ms)
                except PlaywrightTimeoutError:
                    raise BrowserError(
                        f"The text '{value}' did not appear within {wait_s:g}s - the "
                        "page may still be loading or the wording differs. Re-read "
                        "the page, or wait_for with a longer timeout."
                    ) from None
                return {
                    "condition": "text",
                    "text": value,
                    "url": self.active_page.url,
                }

            return await self._run(
                f"Waiting for text '{value}'", op_text, timeout=wait_s + 3
            )

        if cond == "url":
            if not value:
                raise BrowserError(
                    "wait_for('url') needs a fragment the URL should contain."
                )

            async def op_url() -> dict[str, Any]:
                try:
                    await self.active_page.wait_for_url(
                        lambda u: value in str(u), timeout=ms
                    )
                except PlaywrightTimeoutError:
                    raise BrowserError(
                        f"The URL did not contain '{value}' within {wait_s:g}s - the "
                        "page may not have navigated. Check with inspect_page or "
                        "navigate."
                    ) from None
                return {"condition": "url", "url": self.active_page.url}

            return await self._run(
                f"Waiting for URL '{value}'", op_url, timeout=wait_s + 3
            )

        if cond == "load":

            async def op_load() -> dict[str, Any]:
                await self.active_page.wait_for_load_state("load", timeout=ms)
                return {"condition": "load", "url": self.active_page.url}

            return await self._run(
                "Waiting for the page to load", op_load, timeout=wait_s + 3
            )

        if cond == "download":

            async def op_download() -> dict[str, Any]:
                if not self._downloads:
                    try:
                        await asyncio.wait_for(
                            self._download_arrived.wait(), timeout=wait_s
                        )
                    except asyncio.TimeoutError:
                        raise BrowserError(
                            f"No download started within {wait_s:g}s. Click the "
                            "download link first, or pass a longer timeout."
                        ) from None
                if not self._downloads:
                    raise BrowserError("The download was cancelled before it finished.")
                download = self._downloads.popleft()
                if not self._downloads:
                    self._download_arrived.clear()
                path = await download.path()
                return {
                    "condition": "download",
                    "filename": download.suggested_filename,
                    "path": str(path),
                }

            return await self._run(
                "Waiting for a download", op_download, timeout=wait_s + 3
            )

        raise BrowserError(
            f"Unknown wait condition '{condition}'. Use 'text', 'url', 'load', "
            "or 'download'."
        )

    # -- tabs -------------------------------------------------------------------

    async def manage_tabs(
        self, action: str, index: int | None = None, url: str | None = None
    ) -> dict[str, Any]:
        act = (action or "").strip().casefold()
        validated_url = validate_http_url(url) if (act == "new" and url) else None
        idx: int | None = None
        if index is not None:
            try:
                idx = int(index)
            except (TypeError, ValueError):
                raise BrowserError(f"index must be a number, got {index!r}.") from None
        if act == "switch" and idx is None:
            raise BrowserError(
                "'switch' needs an index (1-based tab number), e.g. index=2."
            )

        if act == "list":

            async def op_list() -> dict[str, Any]:
                pages = list(self._ctx.pages)
                tabs = []
                for position, page in enumerate(pages, start=1):
                    try:
                        title = await asyncio.wait_for(page.title(), timeout=2)
                    except Exception:
                        title = ""
                    tabs.append({"index": position, "title": title, "url": page.url})
                active = self.active_page
                active_index = pages.index(active) + 1 if active in pages else 1
                return {
                    "action": "list",
                    "count": len(pages),
                    "active_index": active_index,
                    "tabs": tabs,
                }

            return await self._run(
                "Listing tabs",
                op_list,
                timeout=DEFAULT_ACTION_TIMEOUT + 3,
                check_dialog=False,
            )

        if act == "new":

            async def op_new() -> dict[str, Any]:
                page = await self._ctx.new_page()
                self._wire(page)
                if validated_url:
                    await page.goto(
                        validated_url,
                        wait_until="load",
                        timeout=int(DEFAULT_NAV_TIMEOUT * 1000),
                    )
                self._page = page
                with contextlib.suppress(PlaywrightError):
                    await page.bring_to_front()
                return {"action": "new", "index": len(self._ctx.pages), "url": page.url}

            return await self._run(
                "Opening a new tab",
                op_new,
                timeout=DEFAULT_NAV_TIMEOUT + 3,
                check_dialog=False,
            )

        if act == "switch":

            async def op_switch() -> dict[str, Any]:
                pages = list(self._ctx.pages)
                if idx is None or not 1 <= idx <= len(pages):
                    raise BrowserError(
                        f"Tab {idx} does not exist - {len(pages)} tab(s) are open."
                    )
                page = pages[idx - 1]
                self._page = page
                with contextlib.suppress(PlaywrightError):
                    await page.bring_to_front()
                try:
                    title = await asyncio.wait_for(page.title(), timeout=2)
                except Exception:
                    title = ""
                return {
                    "action": "switch",
                    "index": idx,
                    "title": title,
                    "url": page.url,
                }

            return await self._run(
                f"Switching to tab {idx}",
                op_switch,
                timeout=DEFAULT_ACTION_TIMEOUT + 3,
                check_dialog=False,
            )

        if act == "close":

            async def op_close() -> dict[str, Any]:
                pages = list(self._ctx.pages)
                if idx is None:
                    victim = self.active_page
                elif 1 <= idx <= len(pages):
                    victim = pages[idx - 1]
                else:
                    raise BrowserError(
                        f"Tab {idx} does not exist - {len(pages)} tab(s) are open."
                    )
                closed_index = pages.index(victim) + 1 if victim in pages else 0
                was_active = victim is self._page
                await victim.close()
                if was_active:
                    remaining = list(self._ctx.pages)
                    if remaining:
                        self._page = remaining[0]
                        with contextlib.suppress(PlaywrightError):
                            await remaining[0].bring_to_front()
                    else:
                        self._page = await self._ctx.new_page()
                        self._wire(self._page)
                return {
                    "action": "close",
                    "closed_index": closed_index,
                    "count": len(self._ctx.pages),
                }

            return await self._run(
                f"Closing tab {idx if idx is not None else 'active'}",
                op_close,
                timeout=DEFAULT_ACTION_TIMEOUT + 3,
                check_dialog=False,
            )

        raise BrowserError(
            f"Unknown tab action '{action}'. Use: list, new, switch, close."
        )

    # -- power tools -----------------------------------------------------------

    async def execute_javascript(self, code: str) -> dict[str, Any]:
        async def op() -> dict[str, Any]:
            try:
                value = await self.active_page.evaluate(code)
            except PlaywrightTimeoutError as error:
                raise BrowserError(
                    f"JavaScript timed out: {_first(error)} The page may be busy or "
                    "blocked by a dialog."
                ) from error
            except PlaywrightError as error:
                raise BrowserError(
                    f"JavaScript failed: {_first(error)} Script must be a valid "
                    "expression that returns a simple value (string, number, object)."
                ) from error
            if isinstance(value, str):
                value = _clip(value, 4000)
            return {"result": value}

        return await self._run(
            "Running JavaScript", op, timeout=DEFAULT_ACTION_TIMEOUT + 3
        )

    async def manage_site_data(
        self, action: str, domain: str | None = None
    ) -> dict[str, Any]:
        act = (action or "").strip().casefold()

        if act == "list_cookies":

            async def op_list_cookies() -> dict[str, Any]:
                cookies = await self._ctx.cookies()
                if domain:
                    cookies = [
                        entry for entry in cookies if domain in entry.get("domain", "")
                    ]
                return {
                    "action": act,
                    "cookies": [
                        {
                            "name": entry.get("name", ""),
                            "value": _clip(str(entry.get("value", "")), 200),
                            "domain": entry.get("domain", ""),
                            "path": entry.get("path", ""),
                        }
                        for entry in cookies
                    ],
                }

            return await self._run(
                "Listing cookies", op_list_cookies, timeout=DEFAULT_ACTION_TIMEOUT + 3
            )

        if act == "clear_cookies":

            async def op_clear_cookies() -> dict[str, Any]:
                if domain:
                    await self._ctx.clear_cookies(domain=domain)
                else:
                    await self._ctx.clear_cookies()
                return {"action": act, "cookies": []}

            return await self._run(
                "Clearing cookies",
                op_clear_cookies,
                timeout=DEFAULT_ACTION_TIMEOUT + 3,
            )

        if act == "list_storage":

            async def op_list_storage() -> dict[str, Any]:
                store = await self.active_page.evaluate(_STORAGE_LIST_JS)
                return {
                    "action": act,
                    "storage": dict(store) if isinstance(store, dict) else {},
                }

            return await self._run(
                "Listing site storage",
                op_list_storage,
                timeout=DEFAULT_ACTION_TIMEOUT + 3,
            )

        if act == "clear_storage":

            async def op_clear_storage() -> dict[str, Any]:
                await self.active_page.evaluate(_STORAGE_CLEAR_JS)
                return {"action": act}

            return await self._run(
                "Clearing site storage",
                op_clear_storage,
                timeout=DEFAULT_ACTION_TIMEOUT + 3,
            )

        raise BrowserError(
            f"Unknown action '{action}'. Use: list_cookies, clear_cookies, "
            "list_storage, clear_storage."
        )

    async def handle_dialog(
        self, action: str, text: str | None = None
    ) -> dict[str, Any]:
        """Accept or dismiss the pending dialog (the only way they get answered)."""
        act = (action or "").strip().casefold()
        if act not in ("accept", "dismiss"):
            raise BrowserError("action must be 'accept' or 'dismiss'.")
        async with self._lock:
            dialog = self._pending
            if dialog is None:
                raise BrowserError(
                    "No dialog is currently open, so there is nothing to accept "
                    "or dismiss."
                )
            self._pending = None
            self._dialog_arrived.clear()
            message = dialog.message
            kind = dialog.type
            try:
                if act == "accept":
                    if text is not None:
                        await dialog.accept(text)
                    else:
                        await dialog.accept()
                else:
                    await dialog.dismiss()
            except PlaywrightError as error:
                # The dialog vanished (navigation/close); it is gone either way.
                logger.debug("dialog %s failed: %s", act, error)
            return {"action": act, "message": message, "dialog_type": kind}

    async def network_log(
        self, url_filter: str | None = None, status: int | None = None
    ) -> dict[str, Any]:
        """Recent requests from the in-memory ring buffer (no page access)."""
        entries = [dict(entry) for entry in self._network]
        if url_filter:
            needle = url_filter.casefold()
            entries = [entry for entry in entries if needle in entry["url"].casefold()]
        if status is not None:
            try:
                wanted = int(status)
            except (TypeError, ValueError):
                raise BrowserError(
                    f"status must be a number like 404, got {status!r}."
                ) from None
            entries = [entry for entry in entries if entry["status"] == wanted]
        return {"requests": entries}
