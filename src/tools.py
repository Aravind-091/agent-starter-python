"""Speech-facing browser tools for Jarvis: a focused 19-tool set.

Thin wrappers over :class:`browser.BrowserManager`. Every browser failure is
surfaced as a ``ToolError`` carrying a recovery hint (never a raw traceback).
Consequential actions are gated behind an explicit, single-use user approval
recorded by ``confirm_browser_action``; ``execute_javascript`` and
``upload_file`` always require an explicit ``confirmed=true``.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Awaitable
from typing import Any
from urllib.parse import quote_plus

from livekit.agents import RunContext, function_tool
from livekit.agents.llm import FunctionTool, ImageContent, ToolError

from browser import BrowserError, BrowserManager, validate_http_url

# Latency fillers: speak while the page works, so the user never hears silence.
_FILLER_OPEN = "One moment, Sir. Fetching the page."
_FILLER_WAIT = "One moment, Sir. Waiting for the page to finish loading."
_FILLER_SEARCH = "Searching DuckDuckGo, Sir."

# Targets containing these words are consequential: click/type behind the gate.
_RISKY_ACTION_WORDS = frozenset(
    {
        "accept",
        "approve",
        "buy",
        "checkout",
        "comment",
        "confirm",
        "delete",
        "install",
        "order",
        "pay",
        "payment",
        "post",
        "publish",
        "purchase",
        "register",
        "remove",
        "reply",
        "send",
        "sign",
        "submit",
        "subscribe",
        "transfer",
        "uninstall",
        "unsubscribe",
    }
)

# Free text is only gated on submit, and only for these hard-consequence words.
_STRICT_SUBMIT_WORDS = frozenset(
    {"buy", "purchase", "checkout", "delete", "transfer", "pay", "order"}
)

_WORD_RE = re.compile(r"[a-z]+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").casefold()))


def duckduckgo_search_url(query: str) -> str:
    """Build a DuckDuckGo search URL for a spoken query."""
    cleaned = (query or "").strip()
    if not cleaned:
        raise ValueError("query must not be empty")
    return f"https://duckduckgo.com/?q={quote_plus(cleaned)}"


def _named(tool: FunctionTool) -> FunctionTool:
    """Surface the tool's name as ``tool.name``.

    The SDK exposes it only via ``tool.info.name``/``tool.id``; the tool
    inventory check reads ``.name``, so stamp it onto each bound instance.
    """
    tool.name = tool.info.name  # type: ignore[attr-defined]
    return tool


class BrowserTools:
    """The 19 browser tools handed to the agent as ``Toolset(id="browser")``."""

    def __init__(self, browser: BrowserManager) -> None:
        self.browser = browser
        self._confirmed: set[tuple[str, str]] = set()

    @property
    def tools(self) -> list[FunctionTool]:
        return [
            _named(self.open_url),
            _named(self.search_the_web),
            _named(self.navigate),
            _named(self.read_page),
            _named(self.inspect_page),
            _named(self.take_screenshot),
            _named(self.click),
            _named(self.type_text),
            _named(self.press_key),
            _named(self.scroll),
            _named(self.select_option),
            _named(self.wait_for),
            _named(self.manage_tabs),
            _named(self.execute_javascript),
            _named(self.manage_site_data),
            _named(self.upload_file),
            _named(self.handle_dialog),
            _named(self.network_log),
            _named(self.confirm_browser_action),
        ]

    # -- safety gate ---------------------------------------------------------

    def _require_confirmation(self, action: str, target: str) -> None:
        """Consume the single-use (action, target) approval, or refuse."""
        key = (action.casefold(), target.strip().casefold())
        if key in self._confirmed:
            self._confirmed.discard(key)
            return
        raise ToolError(
            f"'{target}' looks consequential. Explain what {action} on it will do, "
            f"ask the user to confirm, and only after a clear yes call "
            f"confirm_browser_action(action='{action}', target='{target}'), then "
            "retry this exact call."
        )

    async def _run(self, awaitable: Awaitable[Any]) -> Any:
        try:
            return await awaitable
        except BrowserError as error:
            raise ToolError(str(error)) from error

    # -- navigation and reading ------------------------------------------------

    @function_tool()
    async def open_url(self, context: RunContext, url: str) -> dict[str, Any]:
        """Open a website in the browser. Use when the user names a specific site,
        service, or URL; append https:// if no scheme is given. For a general
        lookup with no named destination, use search_the_web instead."""
        try:
            target = validate_http_url(url)
        except BrowserError as error:
            raise ToolError(str(error)) from error
        context.session.say(_FILLER_OPEN)
        return await self._run(self.browser.open_url(target))

    @function_tool()
    async def search_the_web(self, context: RunContext, query: str) -> dict[str, Any]:
        """Search DuckDuckGo for a general question with no named destination (for
        example, the current weather in London). Read results with read_page or
        inspect_page, then open_url the best match if more detail is needed."""
        try:
            url = duckduckgo_search_url(query)
        except ValueError as error:
            raise ToolError(str(error)) from error
        context.session.say(_FILLER_SEARCH)
        return await self._run(self.browser.open_url(url))

    @function_tool()
    async def navigate(self, context: RunContext, action: str) -> dict[str, Any]:
        """Go back, forward, or reload the current page. action must be 'back',
        'forward', or 'reload'."""
        return await self._run(self.browser.navigate(action))

    @function_tool()
    async def read_page(self, context: RunContext) -> dict[str, Any]:
        """Get the visible text of the current page plus its title and URL. Use to
        answer questions about page content; cheaper than inspect_page when no
        interaction is needed."""
        return await self._run(self.browser.read_page())

    @function_tool()
    async def inspect_page(self, context: RunContext) -> dict[str, Any]:
        """List the current page's interactive elements as role and name pairs
        (buttons, links, fields...). Call this before click, type_text,
        select_option, or upload_file unless you already have an exact name from a
        previous inspection."""
        return await self._run(self.browser.inspect_page())

    @function_tool()
    async def take_screenshot(
        self, context: RunContext, full_page: bool = False, target: str | None = None
    ) -> dict[str, Any]:
        """Capture the page (or one element) as an image and attach it to the
        conversation so you can see it. Use after opening or changing a page when
        you must verify what is rendered, find an element, or answer what a page
        looks like."""
        png = await self._run(
            self.browser.take_screenshot(full_page=full_page, target=target)
        )
        encoded = base64.b64encode(png).decode("ascii")
        image = ImageContent(image=f"data:image/png;base64,{encoded}")
        context.session.history.add_message(
            role="user",
            content=["[Screenshot of the current browser page]", image],
        )
        return {
            "captured": True,
            "result": "Screenshot attached - look at the image now to see the page.",
        }

    # -- interacting -------------------------------------------------------------

    @function_tool()
    async def click(
        self,
        context: RunContext,
        target: str,
        action: str = "click",
        force: bool = False,
    ) -> dict[str, Any]:
        """Click, double-click, right-click, or hover an element by its exact name
        from inspect_page. action is 'click' (default), 'double_click',
        'right_click', or 'hover'. Consequential targets (submit, order, delete,
        send...) need prior confirm_browser_action approval."""
        if _words(target) & _RISKY_ACTION_WORDS:
            self._require_confirmation(action, target)
        context.disallow_interruptions()
        return await self._run(self.browser.click(target, action=action, force=force))

    @function_tool()
    async def type_text(
        self,
        context: RunContext,
        target: str,
        text: str,
        submit: bool = False,
    ) -> dict[str, Any]:
        """Fill a text field by its exact name from inspect_page and optionally
        submit the form with Enter (submit=true). Typing into consequential fields,
        or submitting text with order/delete/payment words, requires prior
        confirm_browser_action approval."""
        risky_target = bool(_words(target) & _RISKY_ACTION_WORDS)
        risky_submission = submit and bool(_words(text) & _STRICT_SUBMIT_WORDS)
        if risky_target or risky_submission:
            self._require_confirmation("type_text", target)
        context.disallow_interruptions()
        return await self._run(self.browser.type_text(target, text, submit=submit))

    @function_tool()
    async def press_key(self, context: RunContext, key: str) -> dict[str, Any]:
        """Press a keyboard key or combo on the page: Tab, Enter, Escape,
        ArrowDown, PageUp, Backspace, Space, F5, or single letters/digits
        optionally with Ctrl/Shift/Alt/Meta (e.g. 'Ctrl+l'). Use for navigation and
        shortcuts, not for typing text."""
        context.disallow_interruptions()
        return await self._run(self.browser.press_key(key))

    @function_tool()
    async def scroll(
        self,
        context: RunContext,
        direction: str,
        amount: int | None = None,
        target: str | None = None,
    ) -> dict[str, Any]:
        """Scroll the page up or down by pixels (direction 'up' or 'down', default
        amount 600), or pass an element name as target to scroll it into view. Use
        read_page first to decide how far to scroll."""
        return await self._run(self.browser.scroll(direction, amount, target=target))

    @function_tool()
    async def select_option(
        self, context: RunContext, target: str, option: str
    ) -> dict[str, Any]:
        """Choose an option in a standard dropdown by the element's exact name and
        the option's visible text (e.g. 'Blue'). For custom dropdown menus, click
        the field, then click the option."""
        context.disallow_interruptions()
        return await self._run(self.browser.select_option(target, option))

    @function_tool()
    async def wait_for(
        self,
        context: RunContext,
        condition: str,
        value: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Wait for the page to reach a state before reading it: condition 'text'
        waits for value to appear, 'url' waits until the URL contains value, 'load'
        waits for load completion, 'download' waits for a finished download
        (returns filename and path). timeout is seconds (default 10). Prefer this
        over re-opening a page or guessing when content loads slowly."""
        context.session.say(_FILLER_WAIT)
        return await self._run(self.browser.wait_for(condition, value, timeout=timeout))

    @function_tool()
    async def manage_tabs(
        self,
        context: RunContext,
        action: str,
        index: int | None = None,
        url: str | None = None,
    ) -> dict[str, Any]:
        """Work with browser tabs. action: 'list' returns tabs with 1-based
        indices, 'new' opens a tab (optionally at url), 'switch' focuses a tab by
        index, 'close' closes a tab by index (or the active tab if omitted). Use
        tabs when the user wants two sites open at once."""
        return await self._run(self.browser.manage_tabs(action, index, url))

    # -- power tools (explicit permission) ---------------------------------------

    @function_tool()
    async def execute_javascript(
        self, context: RunContext, code: str, confirmed: bool = False
    ) -> dict[str, Any]:
        """Run a JavaScript expression on the page and return its result. Powerful
        and risky: explain what the code will do, get the user's explicit
        permission, then call again with confirmed=true."""
        if not confirmed:
            raise ToolError(
                "execute_javascript needs explicit permission. Describe what the "
                "code will change and ask the user; after a clear yes, retry with "
                "confirmed=true."
            )
        context.disallow_interruptions()
        return await self._run(self.browser.execute_javascript(code))

    @function_tool()
    async def manage_site_data(
        self, context: RunContext, action: str, domain: str | None = None
    ) -> dict[str, Any]:
        """Inspect or clear site data. action: 'list_cookies' or 'clear_cookies'
        (optional domain filter), 'list_storage' or 'clear_storage' for the current
        page's local storage. Rarely needed; mainly for logouts or stuck pages."""
        return await self._run(self.browser.manage_site_data(action, domain))

    @function_tool()
    async def upload_file(
        self, context: RunContext, target: str, path: str, confirmed: bool = False
    ) -> dict[str, Any]:
        """Attach a local file to a file input named by inspect_page (path is on
        this machine). Sensitive: say which file goes to which site and get the
        user's explicit permission, then call again with confirmed=true."""
        if not confirmed:
            raise ToolError(
                "upload_file needs explicit permission. State the file path and the "
                "destination site, ask the user, and after a clear yes retry with "
                "confirmed=true."
            )
        context.disallow_interruptions()
        return await self._run(self.browser.upload_file(target, path, force=True))

    # -- dialogs and diagnostics --------------------------------------------------

    @function_tool()
    async def handle_dialog(
        self, context: RunContext, action: str, text: str | None = None
    ) -> dict[str, Any]:
        """Respond to an open browser alert, confirm, or prompt: action is 'accept'
        or 'dismiss' (text answers prompts). A tool error mentioning an open dialog
        means call this first, then retry the failed action. Tell the user what the
        dialog said if it matters."""
        return await self._run(self.browser.handle_dialog(action, text))

    @function_tool()
    async def network_log(
        self,
        context: RunContext,
        url_filter: str | None = None,
        status: int | None = None,
    ) -> dict[str, Any]:
        """List recent network requests, optionally filtered by url substring
        and/or HTTP status (e.g. 404), to diagnose a page that will not load.
        Mostly for debugging."""
        return await self._run(self.browser.network_log(url_filter, status))

    @function_tool()
    async def confirm_browser_action(
        self, context: RunContext, action: str, target: str
    ) -> dict[str, Any]:
        """Record the user's explicit approval for ONE consequential browser
        action, then immediately retry the gated call. Call only after the user has
        clearly said yes to exactly this action and target; the approval is
        single-use."""
        key = ((action or "").strip().casefold(), (target or "").strip().casefold())
        if not all(key):
            raise ToolError(
                "confirm_browser_action needs both action and target - copy them "
                "from the gated call's error message."
            )
        self._confirmed.add(key)
        return {
            "confirmed": True,
            "action": key[0],
            "target": key[1],
            "result": "User approval recorded. Retry the gated call now.",
        }
