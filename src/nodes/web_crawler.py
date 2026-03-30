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
    
    # Extract business name for logging
    business_name = state.get("business_name", "Unknown")
    logger.info(f"\n----------------------------------------\n[{business_name}] === Node 2: Web Crawler & Signal Extraction ===")

    execution_log = state.get("execution_log", [])
    website_markdown = None
    extracted_signals = None
    website_crawl_success = False

    try:
        # Get website URL from state or registry data
        website_url = state.get("website_url")
        registry_data = state.get("registry_data")

        if not website_url and registry_data and registry_data.official_website_url:
            website_url = registry_data.official_website_url
            
        if not website_url:
            raise ValueError("No valid website URL found in State or Registry Data. Cannot crawl.")

        logger.info(f"[{business_name}] Crawling website: {website_url}")
        execution_log.append(f"Crawling website: {website_url}")

        # Crawl website and convert to Markdown
        crawl4ai_service = await get_crawl4ai_service()
        website_markdown = await crawl4ai_service.crawl_and_convert(website_url)

        if not website_markdown:
            raise ValueError(f"Failed to crawl {website_url}")

        execution_log.append(f"Website successfully converted to Markdown ({len(website_markdown)} chars)")
        logger.info(f"[{business_name}] Website Markdown extracted ({len(website_markdown)} characters)")

        # Extract signals using LLM with strict Pydantic validation
        llm_service = await get_llm_service()

        extraction_prompt = f"""
Analyze the following website content for hidden business signals AND VERIFY THE INDUSTRY.

CRITICAL: Industry Verification - HVAC SUPPLY EQUALS HVAC DISTRIBUTOR
- This search is specifically for HVAC distribution businesses (HVAC wholesale/distribution companies).
- In this industry, "HVAC Supply", "HVAC Supplier", "HVAC Parts Supplier", "HVAC Wholesaler", and "HVAC Distributor" are SYNONYMOUS TERMS.
- Determine if this website belongs to the HVAC distribution industry by looking for any of these indicators:
  * Business name/description includes: "HVAC", "heating", "cooling", "air conditioning", "AC"
  * AND they mention: "supply", "supplier", "distributor", "wholesale", "parts", "equipment"
- REJECT ONLY if the business is: labor union, government agency, residential HVAC contractor, manufacturer, or completely non-HVAC.
- ACCEPT if the website indicates they are an HVAC supply/wholesale company serving contractors and businesses (they ARE a distributor).
- Look for keywords like: "HVAC supplies", "heating cooling", "air conditioning supplier", "HVAC wholesale", "duct", "compressor", "refrigerant", "furnace", "HVAC equipment", "ductless", "heat pump", "contractor supply", "wholesale distributor", etc.
- SEMANTIC SYNONYMS: In this industry, 'HVAC Supply', 'HVAC Parts Supplier', and 'HVAC Wholesaler' mean the EXACT SAME THING as an HVAC Distributor. If the website indicates they are a supply company selling HVAC equipment, ACCEPT them.

Look for these business signals:
1. E-commerce store presence (Shopify, WooCommerce, custom platform, none)
2. Legacy software mentions (Flash, old ASP.NET, outdated technology)
3. Succession planning signals (family members in business, ownership transition discussions)
4. Owner retirement mentions (age references, retirement timeline)
5. Business size indicators from content
6. Contact information and owner details
7. Team size indicators
8. Domain Redirect: If this page clearly states the website has moved to a new domain, extract the new URL.

Website Content (Markdown):
---
{website_markdown}
---

For each signal, provide:
- Detected (true/false)
- Confidence (0.0-1.0)
- Direct evidence from the text

INDUSTRY DETERMINATION:
- is_target_industry: Set to true if the website clearly operates in HVAC supply/distribution. Set to false ONLY if they are NOT in HVAC at all.
- industry_evidence: Quote or describe specific phrases from the website that justify your determination. Examples:
  * If HVAC: "Website is a family-owned HVAC supply company providing equipment to contractors and businesses"
  * If HVAC: "Founded in 2010 as an HVAC parts supplier and distributor for Ohio and surrounding regions"
  * If NOT HVAC: "Website is for a labor union, not an HVAC business"
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
            logger.warning(f"[{business_name}] Signal extraction returned None")
            website_crawl_success = False

    except Exception as e:
        logger.error(f"[{business_name}] Node 2 error: {e}", exc_info=True)
        execution_log.append(f"Node 2 error: {e}")
        website_crawl_success = False

    elapsed = time.time() - start_time
    logger.info(f"[{business_name}] Node 2 completed in {elapsed:.2f}s")

    return {
        "website_markdown": website_markdown,
        "extracted_signals": extracted_signals,
        "website_crawl_success": website_crawl_success,
        "website_crawl_error": None if website_crawl_success else "Crawl or extraction failed",
        "execution_log": execution_log,
        "node_timestamps": {**state.get("node_timestamps", {}), "web_crawler": elapsed},
    }
