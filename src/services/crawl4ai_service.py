"""Headless browser crawl: Playwright fetch + HTML to Markdown for LLM extraction.

Crawl4AI (the ``crawl4ai`` package) targets Python 3.10+; this service uses Playwright
directly so Python 3.9 environments still get real page content. When ``USE_MOCKS`` is
true, returns deterministic mock markdown for tests.
"""

import asyncio
import logging
import re
from typing import Optional

from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_markdown
from playwright.async_api import async_playwright

from src.config import Config

logger = logging.getLogger(__name__)

# Keep prompts bounded; most signal content is above the fold.
_MAX_MARKDOWN_CHARS = 120_000
_NAV_TIMEOUT_MS = 55_000
_POST_LOAD_WAIT_S = 1.5


def _normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        u = f"https://{u}"
    return u


def _html_to_markdown(html: str) -> str:
    """Strip chrome-heavy tags, prefer main/article/body, convert to Markdown."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "template", "iframe"]):
        tag.decompose()
    root = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
    )
    if not root:
        root = soup
    md = html_to_markdown(
        str(root),
        heading_style="ATX",
        bullets="-",
        strip=["img"],
    )
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


class Crawl4AIService:
    """
    Fetches a URL with headless Chromium and returns Markdown suitable for the LLM.

    Mock mode (``Config.USE_MOCKS``) returns fixed sample content for unit-style runs.
    """

    def __init__(self):
        pass

    async def crawl_and_convert(self, url: str) -> Optional[str]:
        """
        Crawl ``url`` and return Markdown, or ``None`` on hard failure.

        When ``USE_MOCKS`` is true, returns mock Markdown without hitting the network.
        """
        try:
            url = _normalize_url(url)
            logger.info(f"Crawling website: {url}")

            if Config.USE_MOCKS:
                logger.debug("USE_MOCKS=true — returning mock markdown (no browser)")
                return self._mock_markdown(url)

            markdown = await self._crawl_playwright(url)
            if markdown:
                if len(markdown) > _MAX_MARKDOWN_CHARS:
                    markdown = markdown[:_MAX_MARKDOWN_CHARS] + "\n\n…(truncated for token limit)"
                logger.info(f"Crawl succeeded: {len(markdown)} chars of markdown from {url}")
                return markdown

            logger.error(f"Crawl produced empty markdown for {url}")
            return None

        except Exception as e:
            logger.error(f"Crawl failed for {url}: {e}", exc_info=True)
            return None

    async def _crawl_playwright(self, url: str) -> Optional[str]:
        html: Optional[str] = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        user_agent=(
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                        viewport={"width": 1365, "height": 900},
                        java_script_enabled=True,
                    )
                    page = await context.new_page()
                    await page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=_NAV_TIMEOUT_MS,
                    )
                    await asyncio.sleep(_POST_LOAD_WAIT_S)
                    html = await page.content()
                finally:
                    await browser.close()
        except Exception as e:
            err = str(e)
            if "Executable doesn't exist" in err:
                logger.error(
                    "Playwright Chromium is not installed. From your project venv run: "
                    "playwright install chromium"
                )
            else:
                logger.error(f"Playwright error for {url}: {e}", exc_info=True)
            return None

        if not html or len(html.strip()) < 50:
            return None
        return _html_to_markdown(html)

    def _mock_markdown(self, url: str) -> str:
        """Return hardcoded mock Markdown for testing."""
        return f"""# Example Company Corp

## About Us

Example Company Corp is a family-owned HVAC supply company serving Ohio and surrounding regions since 2010.

**Business Address:** 1500 Industrial Drive, Cleveland, OH 44114
**Phone:** (216) 555-0123
**Email:** sales@example-company-corp.test

## Our Services

- HVAC equipment wholesale distribution
- Commercial and residential supplies
- Technical support and consultation

## Company Information

- **Founded:** 2010
- **Owner:** Jordan Lee (Age 62, considering retirement in next 5 years)
- **Employees:** 12-15 staff members
- **Status:** Actively operating

## Technology & Systems

Our ordering system runs on legacy ASP.NET (built 2008), which limits integration capabilities. We are exploring modernization options.

## Succession Planning

Jordan mentions in a recent LinkedIn post that a family member is learning the business and may take over operations. No e-commerce store currently exists - all sales handled through phone and email.

## Contact Information

- **Owner/CEO:** Jordan Lee
- **Sales Contact:** sales@example-company-corp.test
- **LinkedIn:** linkedin.com/company/example-company-corp

---

*Last updated: {url}*
"""


_crawl4ai_service: Optional[Crawl4AIService] = None


async def get_crawl4ai_service() -> Crawl4AIService:
    """Get or initialize the global crawler service."""
    global _crawl4ai_service
    if _crawl4ai_service is None:
        _crawl4ai_service = Crawl4AIService()
    return _crawl4ai_service
