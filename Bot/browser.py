
import time
from patchright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from HTMLContentExtractor import HTMLContentExtractor

def fetch_full_html(url: str, timeout: int = 60000, headless: bool = True) -> str:
    """
    Launches a headless Chromium browser, navigates to `url`,
    waits for the network to be idle, and returns the rendered HTML.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/113.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            java_script_enabled=True,
        )
        page = ctx.new_page()
        page.set_default_navigation_timeout(timeout)
        try:
            # Navigate and wait until network is idle (no more than 0.5s of outstanding requests)
            page.goto(url, wait_until="networkidle")
            # Extra buffer for late JS-rendered changes 
            time.sleep(5)
            return page.content()
        except PlaywrightTimeoutError:
            # On timeout, return whatever has loaded so far
            return page.content()
        finally:
            browser.close()

if __name__ == "__main__":
    URL = "https://www.gjopen.com/questions/4349-before-1-january-2026-will-the-us-senate-pass-reconciliation-legislation-that-includes-a-moratorium-on-state-and-local-enforcement-of-regulations-regarding-artificial-intelligence"
    html = fetch_full_html(URL)
    extractor = HTMLContentExtractor()
    result = extractor.extract(URL, html)
    print(html)
    print("Markdown: ")
    print(result)