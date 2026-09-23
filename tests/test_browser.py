"""Behavior tests for the Playwright-backed BrowserManager.

Run with: uv run pytest tests/test_browser.py
Requires: uv run playwright install chromium
"""

import asyncio
from pathlib import Path

import pytest

from browser import BrowserError, BrowserManager


async def wait_for_dialog(browser: BrowserManager) -> dict:
    """Dialog events land asynchronously after the click that opened them."""
    for _ in range(300):
        pending = browser.pending_dialog
        if pending is not None:
            return pending
        await asyncio.sleep(0.01)
    raise AssertionError("no dialog was recorded")


async def test_read_page_returns_visible_text(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    result = await browser.read_page()
    assert "Jarvis browser test page" in result["text"]
    assert result["title"] == "Test Page"


async def test_inspect_page_lists_interactive_elements(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    result = await browser.inspect_page()
    names = [el["name"] for el in result["elements"]]
    roles = {el["role"] for el in result["elements"]}
    assert "Add to cart" in names
    assert "Full name" in names
    assert "Favorite color" in names
    assert "button" in roles
    assert "link" in roles


async def test_click_by_accessible_name(browser: BrowserManager, page_url: str) -> None:
    await browser.open_url(page_url)
    result = await browser.click("Add to cart")
    assert result["clicked"] == "Add to cart"
    state = await browser.execute_javascript(
        "document.getElementById('count').textContent"
    )
    assert state["result"] == "1"


async def test_click_double_and_right(browser: BrowserManager, page_url: str) -> None:
    await browser.open_url(page_url)
    await browser.click("Double Me", action="double_click")
    state = await browser.execute_javascript(
        "document.getElementById('dbl-count').textContent"
    )
    assert state["result"] == "2"

    await browser.click("Right Target", action="right_click")
    state = await browser.execute_javascript(
        "document.getElementById('right-result').textContent"
    )
    assert state["result"] == "right-clicked"


async def test_click_hover_does_not_raise(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    result = await browser.click("Add to cart", action="hover")
    assert result["hovered"] == "Add to cart"


async def test_click_missing_target_raises(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    with pytest.raises(BrowserError, match="Could not find"):
        await browser.click("Nonexistent Widget")


async def test_type_text_into_labeled_field(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    await browser.type_text("Full name", "Jarvis")
    state = await browser.execute_javascript(
        "document.getElementById('fullname').value"
    )
    assert state["result"] == "Jarvis"


async def test_type_text_submit_sends_form(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    await browser.type_text("Full name", "Sir Reginald", submit=True)
    state = await browser.execute_javascript(
        "document.getElementById('submitted').textContent"
    )
    assert state["result"] == "Sir Reginald"


async def test_select_option(browser: BrowserManager, page_url: str) -> None:
    await browser.open_url(page_url)
    await browser.select_option("Favorite color", "Blue")
    state = await browser.execute_javascript("document.getElementById('color').value")
    assert state["result"] == "b"


async def test_press_key_tab_moves_focus(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    await browser.press_key("Tab")
    await browser.press_key("Tab")
    focused = await browser.execute_javascript("document.activeElement.id")
    assert focused["result"] != ""


async def test_press_key_rejects_unknown_key(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    with pytest.raises(BrowserError, match=r"not allowed|Unknown key|not supported"):
        await browser.press_key("MegaKey")


async def test_scroll_down_changes_offset(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    await browser.scroll("down", amount=800)
    offset = await browser.execute_javascript("window.scrollY")
    assert offset["result"] >= 700


async def test_scroll_up_returns_to_top(browser: BrowserManager, page_url: str) -> None:
    await browser.open_url(page_url)
    await browser.scroll("down", amount=800)
    await browser.scroll("up", amount=800)
    offset = await browser.execute_javascript("window.scrollY")
    assert offset["result"] == 0


async def test_wait_for_text_waits_for_delayed_content(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    result = await browser.wait_for("text", "Delayed content", timeout=5)
    assert result["condition"] == "text"


async def test_wait_for_text_times_out_with_helpful_error(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    with pytest.raises(BrowserError, match=r"did not appear|timed out|Timeout"):
        await browser.wait_for("text", "Never Rendered", timeout=1)


async def test_wait_for_url(browser: BrowserManager, page_url: str) -> None:
    await browser.open_url(page_url)
    await browser.click("About")
    result = await browser.wait_for("url", "about.html", timeout=5)
    assert "about.html" in result["url"]


async def test_tabs_lifecycle(browser: BrowserManager, page_url: str) -> None:
    await browser.open_url(page_url)
    await browser.manage_tabs("new")
    listing = await browser.manage_tabs("list")
    assert listing["count"] == 2
    assert listing["active_index"] == 2

    await browser.manage_tabs("switch", index=1)
    listing = await browser.manage_tabs("list")
    assert listing["active_index"] == 1
    assert "Test Page" in listing["tabs"][0]["title"]

    await browser.manage_tabs("close", index=1)
    listing = await browser.manage_tabs("list")
    assert listing["count"] == 1


async def test_execute_javascript_returns_value(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    result = await browser.execute_javascript("6 * 7")
    assert result["result"] == 42
    titled = await browser.execute_javascript("document.title")
    assert titled["result"] == "Test Page"


async def test_execute_javascript_reports_page_errors(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    with pytest.raises(BrowserError):
        await browser.execute_javascript("throw new Error('boom')")


async def test_click_inside_iframe_resolves(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    # The page frame plus the embedded child frame must both be searchable.
    assert len(browser.active_page.frames) > 1
    await browser.click("Iframe Button")
    echo = await browser.execute_javascript(
        "document.getElementById('child').contentDocument.getElementById('iframe-echo').textContent"
    )
    assert echo["result"] == "iframe clicked"


async def test_dialog_is_recorded_then_can_be_accepted(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    await browser.click("Danger Zone", action="click", force=True)
    pending = await wait_for_dialog(browser)
    assert "Delete everything?" in pending["message"]

    await browser.handle_dialog("accept")
    result = await browser.execute_javascript(
        "document.getElementById('dialog-result').textContent"
    )
    assert result["result"] == "accepted"


async def test_dialog_can_be_dismissed(browser: BrowserManager, page_url: str) -> None:
    await browser.open_url(page_url)
    await browser.click("Danger Zone", action="click", force=True)
    await wait_for_dialog(browser)
    await browser.handle_dialog("dismiss")
    result = await browser.execute_javascript(
        "document.getElementById('dialog-result').textContent"
    )
    assert result["result"] == "dismissed"


async def test_handle_dialog_without_dialog_raises(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    with pytest.raises(BrowserError, match=r"No dialog|no dialog"):
        await browser.handle_dialog("accept")


async def test_network_log_records_requests(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    log = await browser.network_log()
    urls = [entry["url"] for entry in log["requests"]]
    assert any("index.html" in url for url in urls)
    missing = await browser.network_log(status=404)
    assert any("missing.css" in entry["url"] for entry in missing["requests"])


async def test_upload_file(browser: BrowserManager, page_url: str, tmp_path) -> None:
    payload = tmp_path / "resume.txt"
    payload.write_text("important words", encoding="utf-8")
    await browser.open_url(page_url)
    await browser.upload_file("upload", str(payload), force=True)
    state = await browser.execute_javascript(
        "document.getElementById('upload-name').textContent"
    )
    assert state["result"] == "resume.txt"


async def test_download_wait_returns_saved_file(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    await browser.click("Download notes", force=True)
    result = await browser.wait_for("download", timeout=5)
    assert result["filename"] == "download.txt"
    assert Path(result["path"]).exists()


async def test_site_data_cookies_roundtrip(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    await browser.execute_javascript("document.cookie = 'session=abc123; path=/'")
    cookies = await browser.manage_site_data("list_cookies")
    assert any(c["name"] == "session" for c in cookies["cookies"])

    await browser.manage_site_data("clear_cookies")
    cookies = await browser.manage_site_data("list_cookies")
    assert all(c["name"] != "session" for c in cookies["cookies"])


async def test_site_data_storage_roundtrip(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    await browser.execute_javascript("localStorage.setItem('cart', 'eggs')")
    storage = await browser.manage_site_data("list_storage")
    assert "cart" in storage["storage"]

    await browser.manage_site_data("clear_storage")
    storage = await browser.manage_site_data("list_storage")
    assert "cart" not in storage["storage"]


async def test_screenshot_returns_png_bytes(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    png = await browser.take_screenshot()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 1000


async def test_navigate_back_and_forward(
    browser: BrowserManager, page_url: str
) -> None:
    await browser.open_url(page_url)
    await browser.click("About")
    await browser.wait_for("url", "about.html", timeout=5)
    await browser.navigate("back")
    await browser.wait_for("text", "Jarvis browser test page", timeout=5)
    await browser.navigate("forward")
    result = await browser.navigate("reload")
    assert result["action"] == "reload"


async def test_open_url_rejects_bad_scheme(browser: BrowserManager) -> None:
    with pytest.raises(BrowserError, match="http"):
        await browser.open_url("javascript:alert(1)")


async def test_close_is_idempotent() -> None:
    manager = BrowserManager(headless=True)
    await manager.close()
    await manager.close()
