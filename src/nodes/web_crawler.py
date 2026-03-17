"""Node 2: Dynamic Website Crawler & Signal Extraction."""

import logging
import time
from src.models.state import LeadState
from src.models.schemas import WebsiteSignals
from src.services.crawl4ai_service import get_crawl4ai_service
from src.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


async def web_crawler_node(state: LeadState) -> dict:
    """
    Node 2: Dynamic Website Crawler & Signal Extraction.

    Uses Crawl4AI (local Playwright) to fetch website and convert to Markdown.
    Uses LLM to extract hidden signals from Markdown with strict Pydantic validation.

    Signals extracted:
    - E-commerce presence
    - Legacy software mentions
    - Succession planning signals
    - Owner information and retirement mentions

    Args:
        state: Current LeadState from graph

    Returns:
        Updated state dict with website signals
    """
    start_time = time.time()
    logger.info("=== Node 2: Web Crawler & Signal Extraction ===")

    execution_log = state.get("execution_log", [])
    website_markdown = None
    extracted_signals = None
    website_crawl_success = False

    try:
        # Get website URL from state or registry data
        website_url = state.get("website_url")
        if not website_url and state.get("registry_data"):
            # Try to derive from business name
            business_name = state.get("business_name", "").lower().replace(" ", "")
            website_url = f"https://www.{business_name}.com"

        if not website_url:
            raise ValueError("No website URL available to crawl")

        logger.info(f"Crawling website: {website_url}")
        execution_log.append(f"Crawling website: {website_url}")

        # Crawl website and convert to Markdown
        crawl4ai_service = await get_crawl4ai_service()
        website_markdown = await crawl4ai_service.crawl_and_convert(website_url)

        if not website_markdown:
            raise ValueError(f"Failed to crawl {website_url}")

        execution_log.append(f"Website successfully converted to Markdown ({len(website_markdown)} chars)")
        logger.info(f"Website Markdown extracted ({len(website_markdown)} characters)")

        # Extract signals using LLM with strict Pydantic validation
        llm_service = await get_llm_service()

        extraction_prompt = f"""
Analyze the following website content for hidden business signals.
Look for:
1. E-commerce store presence (Shopify, WooCommerce, custom platform, none)
2. Legacy software mentions (Flash, old ASP.NET, outdated technology)
3. Succession planning signals (family members in business, ownership transition discussions)
4. Owner retirement mentions (age references, retirement timeline)
5. Business size indicators from content
6. Contact information and owner details
7. Team size indicators

Website Content (Markdown):
---
{website_markdown}
---

For each signal, provide:
- Detected (true/false)
- Confidence (0.0-1.0)
- Direct evidence from the text
"""

        extracted_signals = await llm_service.extract_structured(
            prompt=extraction_prompt,
            response_model=WebsiteSignals,
            context="Extract business signals from website content. Be conservative - only flag signals with strong evidence (0.7+ confidence).",
        )

        if extracted_signals:
            execution_log.append(
                f"Website signals extracted: e-commerce={extracted_signals.has_ecommerce_store.detected}, "
                f"legacy_tech={extracted_signals.legacy_software_mentions.detected}, "
                f"succession={extracted_signals.succession_planning_signals.detected}"
            )
            website_crawl_success = True
        else:
            logger.warning("Signal extraction returned None")
            website_crawl_success = False

    except Exception as e:
        logger.error(f"Node 2 error: {e}", exc_info=True)
        execution_log.append(f"Node 2 error: {e}")
        website_crawl_success = False

    elapsed = time.time() - start_time
    logger.info(f"Node 2 completed in {elapsed:.2f}s")

    return {
        "website_markdown": website_markdown,
        "extracted_signals": extracted_signals,
        "website_crawl_success": website_crawl_success,
        "website_crawl_error": None if website_crawl_success else "Crawl or extraction failed",
        "execution_log": execution_log,
        "node_timestamps": {**state.get("node_timestamps", {}), "web_crawler": elapsed},
    }
