import functools
import http.server
import threading
from pathlib import Path

import pytest

from browser import BrowserManager

FIXTURES = Path(__file__).parent / "fixtures"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


@pytest.fixture(scope="module")
def site_url() -> str:
    """Serve tests/fixtures over local HTTP so cookies and downloads behave."""
    handler = functools.partial(_QuietHandler, directory=str(FIXTURES))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


@pytest.fixture
async def browser() -> BrowserManager:
    manager = BrowserManager(headless=True)
    try:
        yield manager
    finally:
        await manager.close()


@pytest.fixture
def page_url(site_url: str) -> str:
    return f"{site_url}/index.html"
