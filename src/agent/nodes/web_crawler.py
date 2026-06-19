"""Node 2: Dynamic Website Crawler & Signal Extraction."""

import logging
import time
from src.agent.state import LeadState
from src.models.schemas import WebsiteSignals
from src.services.crawl4ai_service import get_crawl4ai_service
from src.services.llm_service import EMBEDDING_MODEL, get_llm_service

logger = logging.getLogger(__name__)


async def web_crawler_node(state: LeadState) -> dict:
    """
    Node 2: Dynamic Website Crawler & Signal Extraction.

    Uses Crawl4AI (local Playwright) to fetch website and convert to Markdown.
    Uses LLM to extract hidden signals from Markdown with strict Pydantic validation.

    Signals extracted:
    - Dynamic list (3–5) aligned to the batch ``investment_thesis`` in state

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
    embedding = None
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

        llm_service = await get_llm_service()

        embedding = await llm_service.generate_embedding(
            website_markdown, business_name=business_name
        )
        if embedding:
            execution_log.append(
                f"Generated {len(embedding)}-dim embedding via {EMBEDDING_MODEL}"
            )
            logger.info(
                f"[{business_name}] Embedding generated ({len(embedding)} dimensions)"
            )
        else:
            execution_log.append("Embedding generation failed or skipped")
            logger.warning(f"[{business_name}] Embedding generation returned None")

        # Extract dynamic state variables or fallback to defaults
        investment_thesis = (state.get("investment_thesis") or "Find strategic operational signals.").strip()
        industry_definition = (state.get("industry_definition") or "Any valid B2B business.").strip()
        location = state.get("location") or ""
        query = (state.get("query") or "").strip()

        # Dynamically inject independence exclusion only when the query requires it
        requires_independence = any(
            kw in query.lower()
            for kw in ["independent", "do not include merged", "not merged", "no mergers"]
        )
        independence_exclusion = ""
        if requires_independence:
            independence_exclusion = (
                "\n5. The SEARCH QUERY requires 'independent' distributors AND any part of the website "
                "(headers, footers, about pages, legal notices, or any visible text) identifies "
                "this company as a subsidiary, division, or joint venture of a larger corporation. "
                "Look specifically at: footer copyright lines (e.g., '© 2024 XYZ, a Watsco Company'), "
                "about-page ownership disclosures (e.g., 'a Ferguson plc company'), "
                "or explicit JV statements (e.g., 'a joint venture between Carrier and Watsco'). "
                "Geographic limitations or small local footprint do NOT trigger this criterion. "
                "Independently-owned franchise locations are NOT excluded."
            )

        extraction_prompt = f"""
Analyze the following website content for hidden business signals AND VERIFY THE TARGET INDUSTRY/TECH STACK.

=== SEARCH QUERY (for context) ===
{query}

=== INVESTMENT / QUALIFICATION THESIS ===
{investment_thesis}

=== INDUSTRY / TECH STACK DEFINITION ===
{industry_definition}

=== EXCLUSION CRITERIA (OVERRIDE ALL OTHER SIGNALS) ===
You MUST set `is_target_industry` to FALSE if ANY of the following conditions are true — regardless of any partial overlap with HVAC terminology:
1. EXCLUDE ONLY if HVAC equipment distribution is completely absent, OR if the company is fundamentally a non-HVAC business that mentions HVAC only in passing (e.g., a pure electrical supply house, a pure waterworks pipeline products company). DO NOT EXCLUDE a company that distributes HVAC equipment and heating/cooling products to contractor customers, even if they also distribute plumbing, hydronic, or pipe/fitting products alongside HVAC. Multi-category B2B wholesale distributors that include HVAC as a real product line — such as companies selling HVAC, plumbing, and hydronics together to trade contractors — ARE valid targets and must NOT be excluded by this criterion.
2. The website is for a SaaS software company, field service management platform, or technology vendor — even if they specifically serve HVAC contractors.
3. The website is for a Manufacturer (i.e., a company that MAKES HVAC equipment such as compressors, chillers, or air handlers). Manufacturers are NOT distributors. A company that designs, engineers, or produces HVAC units is NOT a distributor.
4. The company is a labor union, trade association, training school, or advocacy organization.{independence_exclusion}
Do NOT let mentions of HVAC products, customers, or partnerships override these exclusion rules. The company itself must be an HVAC Distributor (a company that SOURCES and RESELLS HVAC equipment from manufacturers to contractors/dealers).

=== TARGET GEOGRAPHY ===
{location}

CRITICAL GEOGRAPHIC RULE: You must cross-reference the TARGET GEOGRAPHY with the service areas, branches, or HQ locations listed on the website.
- Extract ALL US states they operate in into `operating_states`. Include states mentioned in: branch locations, distribution centers, service areas, "we serve..." statements, regional headings, store finders, or any geographic reference.
- Set `operates_in_target_location` to True if: (a) the site explicitly mentions serving or having locations in the TARGET GEOGRAPHY, OR (b) the company claims multi-state, regional, or nationwide coverage that plausibly includes the TARGET GEOGRAPHY.
- Set `operates_in_target_location` to False ONLY if the website clearly implies a geographic focus that EXCLUDES the TARGET GEOGRAPHY (e.g., "serving the Northeast only" when the target is Florida).
- Leave as null ONLY when the website has no geographic information whatsoever.

Based on this Thesis, you MUST dynamically identify 3 to 5 critical, concrete signals to look for on this website.
For EACH signal, output exactly one object in the JSON field `signals` with: signal_name, detected (bool), confidence (0.0–1.0), and evidence.

CRITICAL NAME EXTRACTION: Extract the ACTUAL business name as it appears in the website markdown. DO NOT use placeholder names, and DO NOT hallucinate.

INDUSTRY / QUALIFICATION DETERMINATION:
- is_target_industry: Set to true ONLY if the website clearly matches the INDUSTRY / TECH STACK DEFINITION provided above AND passes all EXCLUSION CRITERIA above.
- industry_evidence: Quote or describe specific phrases from the website that justify your determination. If excluding, cite which exclusion criterion applies.

Also extract where applicable:
- Business size indicators, team size hints, contact information, owner details
- Domain Redirect: If this page clearly states the website has moved, extract the new URL in new_domain_redirect

Website Content (Markdown):
---
{website_markdown}
---
"""
        extracted_signals = await llm_service.extract_structured(
            prompt=extraction_prompt,
            response_model=WebsiteSignals,
            context=(
                "Extract WebsiteSignals from markdown. Populate `signals` with 3-5 thesis-aligned rows. "
                "Be conservative: only detected=true with solid on-page evidence."
            ),
        )

        if extracted_signals:
            det = [s.signal_name for s in (extracted_signals.signals or []) if s.detected]
            execution_log.append(
                f"Website signals extracted: {len(extracted_signals.signals or [])} thesis rows, "
                f"detected_true={det}"
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
        "embedding": embedding,
        "extracted_signals": extracted_signals,
        "website_crawl_success": website_crawl_success,
        "website_crawl_error": None if website_crawl_success else "Crawl or extraction failed",
        "execution_log": execution_log,
        "node_timestamps": {**state.get("node_timestamps", {}), "web_crawler": elapsed},
    }
