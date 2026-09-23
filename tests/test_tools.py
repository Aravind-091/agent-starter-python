"""Tool-layer tests: confirmation gate, error mapping, session integration.

These use a fake BrowserManager so they run without launching Chromium.
"""

from types import SimpleNamespace

import pytest
from livekit.agents.llm import ToolError

from browser import BrowserError
from tools import BrowserTools, duckduckgo_search_url


class FakeBrowser:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fail_with: BrowserError | None = None

    def _record(self, *args: object) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append(args)

    async def open_url(self, url: str) -> dict:
        self._record(url)
        return {"title": "T", "url": url}

    async def read_page(self) -> dict:
        self._record()
        return {"text": "hello", "title": "T", "url": "u"}

    async def inspect_page(self) -> dict:
        self._record()
        return {"elements": [], "text": "t", "title": "T", "url": "u"}

    async def go_back_equivalent(self) -> None:  # pragma: no cover - placeholder
        pass

    async def navigate(self, action: str) -> dict:
        self._record(action)
        return {"action": action, "url": "u"}

    async def take_screenshot(
        self, full_page: bool = False, target: str | None = None
    ) -> bytes:
        self._record(full_page, target)
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 2000

    async def click(
        self, target: str, action: str = "click", force: bool = False
    ) -> dict:
        self._record(target, action)
        return {"clicked": target}

    async def type_text(self, target: str, text: str, submit: bool = False) -> dict:
        self._record(target, text, submit)
        return {"typed": target}

    async def press_key(self, key: str) -> dict:
        self._record(key)
        return {"key": key}

    async def scroll(
        self, direction: str, amount: int | None = None, target: str | None = None
    ) -> dict:
        self._record(direction, amount, target)
        return {"direction": direction}

    async def select_option(self, target: str, option: str) -> dict:
        self._record(target, option)
        return {"selected": target}

    async def wait_for(
        self, condition: str, value: str | None = None, timeout: float | None = None
    ) -> dict:
        self._record(condition, value, timeout)
        return {"condition": condition, "url": "u"}

    async def manage_tabs(
        self, action: str, index: int | None = None, url: str | None = None
    ) -> dict:
        self._record(action, index, url)
        return {"action": action, "count": 1, "tabs": [], "active_index": 1}

    async def execute_javascript(self, code: str) -> dict:
        self._record(code)
        return {"result": 1}

    async def manage_site_data(self, action: str, domain: str | None = None) -> dict:
        self._record(action, domain)
        return {"action": action, "cookies": [], "storage": {}}

    async def upload_file(self, target: str, path: str, force: bool = False) -> dict:
        self._record(target, path)
        return {"uploaded": path}

    async def handle_dialog(self, action: str, text: str | None = None) -> dict:
        self._record(action, text)
        return {"action": action}

    async def network_log(
        self, url_filter: str | None = None, status: int | None = None
    ) -> dict:
        self._record(url_filter, status)
        return {"requests": []}


class FakeHistory:
    def __init__(self) -> None:
        self.added: list[dict] = []

    def add_message(self, **kwargs: object) -> None:
        self.added.append(kwargs)


def make_context() -> SimpleNamespace:
    said: list[str] = []
    interruptions: list[bool] = []

    def say(text: str) -> None:
        said.append(text)

    def disallow() -> None:
        interruptions.append(True)

    session = SimpleNamespace(say=say)
    ctx = SimpleNamespace(session=session, disallow_interruptions=disallow)
    ctx._said = said  # type: ignore[attr-defined]
    ctx._interruptions = interruptions  # type: ignore[attr-defined]
    ctx.session.history = FakeHistory()  # type: ignore[attr-defined]
    return ctx


@pytest.fixture
def fake_browser() -> FakeBrowser:
    return FakeBrowser()


@pytest.fixture
def tools(fake_browser: FakeBrowser) -> BrowserTools:
    return BrowserTools(fake_browser)  # type: ignore[arg-type]


@pytest.fixture
def ctx() -> SimpleNamespace:
    return make_context()


# --- confirmation gate -------------------------------------------------------


async def test_safe_click_runs_without_confirmation(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    await tools.click(ctx, "Add to cart")
    assert fake_browser.calls == [("Add to cart", "click")]


async def test_risky_click_requires_confirmation(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    with pytest.raises(ToolError, match="confirm"):
        await tools.click(ctx, "Submit Order")
    assert fake_browser.calls == []


async def test_confirmed_click_then_consumes_token(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    await tools.confirm_browser_action(ctx, action="click", target="Submit Order")
    await tools.click(ctx, "Submit Order")
    assert fake_browser.calls == [("Submit Order", "click")]

    # token is single-use
    with pytest.raises(ToolError, match="confirm"):
        await tools.click(ctx, "Submit Order")


async def test_confirmation_is_action_scoped(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    await tools.confirm_browser_action(ctx, action="click", target="Send Message")
    with pytest.raises(ToolError, match="confirm"):
        await tools.type_text(ctx, "Send Message", "hi", submit=True)
    assert fake_browser.calls == []


async def test_risky_submit_typing_requires_confirmation(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    with pytest.raises(ToolError, match="confirm"):
        await tools.type_text(ctx, "Checkout Form", "4111 1111", submit=True)
    assert fake_browser.calls == []


async def test_plain_typing_is_not_gated(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    await tools.type_text(ctx, "Search box", "weather today")
    assert fake_browser.calls == [("Search box", "weather today", False)]


# --- gated power tools -------------------------------------------------------


async def test_javascript_requires_confirmation(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    with pytest.raises(ToolError, match=r"permission|confirm"):
        await tools.execute_javascript(ctx, "document.title")
    assert fake_browser.calls == []


async def test_javascript_allowed_when_confirmed(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    result = await tools.execute_javascript(ctx, "6 * 7", confirmed=True)
    assert result["result"] == 1
    assert ctx._interruptions  # mutating tool disallowed interruptions


async def test_upload_requires_confirmation(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    with pytest.raises(ToolError, match=r"permission|confirm"):
        await tools.upload_file(ctx, "upload", "C:/tmp/a.txt")
    assert fake_browser.calls == []


async def test_upload_allowed_when_confirmed(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    await tools.upload_file(ctx, "upload", "C:/tmp/a.txt", confirmed=True)
    assert fake_browser.calls == [("upload", "C:/tmp/a.txt")]


# --- screenshot -> image in history -----------------------------------------


async def test_screenshot_adds_image_to_history(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    from livekit.agents.llm import ImageContent

    result = await tools.take_screenshot(ctx)
    assert result["captured"] is True
    added = ctx.session.history.added
    assert len(added) == 1
    content = added[0]["content"]
    assert any(isinstance(part, ImageContent) for part in content)
    image_part = next(part for part in content if isinstance(part, ImageContent))
    assert str(image_part.image).startswith("data:image/png;base64,")


# --- error mapping -----------------------------------------------------------


async def test_browser_errors_become_tool_errors(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    fake_browser.fail_with = BrowserError("Page did not load in time.")
    with pytest.raises(ToolError, match="Page did not load"):
        await tools.open_url(ctx, "https://example.com")


async def test_open_url_rejects_non_http(tools: BrowserTools, ctx) -> None:
    with pytest.raises(ToolError, match="http"):
        await tools.open_url(ctx, "file:///etc/passwd")


# --- latency fillers ---------------------------------------------------------


async def test_open_url_speaks_filler(tools: BrowserTools, ctx) -> None:
    await tools.open_url(ctx, "https://example.com")
    assert len(ctx._said) == 1


async def test_wait_for_speaks_filler(tools: BrowserTools, ctx) -> None:
    await tools.wait_for(ctx, "text", "hello")
    assert len(ctx._said) == 1


# --- search ------------------------------------------------------------------


async def test_search_the_web_builds_duckduckgo_url(
    tools: BrowserTools, fake_browser: FakeBrowser, ctx
) -> None:
    await tools.search_the_web(ctx, "current weather in London")
    (called_url,) = fake_browser.calls[0]
    assert called_url.startswith("https://duckduckgo.com/?")
    assert (
        "current+weather+in+London" in called_url or "current%20weather" in called_url
    )


def test_duckduckgo_search_url_rejects_empty() -> None:
    with pytest.raises(ValueError):
        duckduckgo_search_url("   ")


# --- tool inventory ----------------------------------------------------------


def test_toolset_is_consolidated(tools: BrowserTools) -> None:
    """LiveKit guidance: keep the tool list focused (<= ~20 with the gate)."""
    names = [t.info.name for t in tools.tools]
    assert len(names) == 19
    assert len(set(names)) == len(names)
    for expected in (
        "open_url",
        "search_the_web",
        "navigate",
        "read_page",
        "inspect_page",
        "take_screenshot",
        "click",
        "type_text",
        "press_key",
        "scroll",
        "select_option",
        "wait_for",
        "manage_tabs",
        "execute_javascript",
        "manage_site_data",
        "upload_file",
        "handle_dialog",
        "network_log",
        "confirm_browser_action",
    ):
        assert expected in names
    assert "go_back" not in names  # consolidated into navigate
