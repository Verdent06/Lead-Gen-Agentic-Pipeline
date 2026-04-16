"""Node 1: Discovery & State Registry Verification."""

import logging
import time
from src.models.state import LeadState
from src.models.schemas import RegistryVerification, WebsiteDiscovery
from src.services.tavily_service import get_tavily_service
from src.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


async def discovery_node(state: LeadState) -> dict:
    """
    Node 1: Discovery & State Registry Check.

    Searches for official state licensing registries using Tavily.
    Uses LLM to extract structured RegistryVerification from search results.
    Routes based on business status: Active → continue, Inactive/Not Found → END.

    Args:
        state: Current LeadState from graph

    Returns:
        Updated state dict with registry verification results
    """
    start_time = time.time()
    
    # Extract business name for logging
    business_name = state.get("business_name", "Unknown")
    logger.info(f"\n----------------------------------------\n[{business_name}] === Node 1: Discovery & Registry Verification ===")

    execution_log = state.get("execution_log", [])
    registry_search_query = None
    search_results = {}
    registry_data = None
    should_continue = False

    try:
        # Extract input parameters
        query = state.get("query", "")
        location = state.get("location", "")

        if not business_name or not location:
            raise ValueError("business_name and location are required")

        # Construct refined search query
        registry_search_query = f"{business_name} {location} state licensing registry active business"
        logger.info(f"[{business_name}] Searching for: {registry_search_query}")

        # Perform Tavily search
        tavily_service = await get_tavily_service()
        search_results = await tavily_service.search(
            query=registry_search_query,
            include_answer=True,
            num_results=20,
            topic="general",
        )

        execution_log.append(
            f"Node 1: Tavily search completed with {len(search_results.get('results', []))} results"
        )

        # Extract structured data from search results using LLM
        llm_service = await get_llm_service()

        # Format search results as text for LLM
        search_text = f"""
Business: {business_name}
Location: {location}
Query: {query}

Search Results:
{search_results.get('answer', '')}

Details:
"""
        for idx, result in enumerate(search_results.get("results", [])[:5], 1):
            search_text += f"\n{idx}. {result.get('title', '')}\n{result.get('content', '')}"

        # Extract registry verification using LLM
        registry_data = await llm_service.extract_structured(
            prompt=search_text,
            response_model=RegistryVerification,
            context="Extract business registry information from search results. Set confidence high (0.8-1.0) if business is clearly active and found.",
        )

        if registry_data:
            execution_log.append(f"Registry verification: {registry_data.registry_status}")
            logger.info(f"[{business_name}] Business status: {registry_data.registry_status}")

            # Log the registry source URL
            if registry_data.registry_url:
                logger.info(f"[{business_name}] Registry Source URL: {registry_data.registry_url}")
                execution_log.append(f"Registry Source: {registry_data.registry_url}")
            else:
                logger.info(f"[{business_name}] Registry verified via aggregate search context (no direct URL available)")
                execution_log.append("Registry verified via aggregate search context (no direct URL available)")

            # === FALLBACK: Search for website URL if not found in registry search ===
            if not registry_data.official_website_url:
                logger.info(f"[{business_name}] Website URL not found in registry search. Attempting fallback website search...")
                execution_log.append("Website URL missing - attempting fallback search")

                try:
                    # Perform targeted website search
                    website_search_query = f"{business_name} {location} official website"
                    website_search_results = await tavily_service.search(
                        query=website_search_query,
                        include_answer=True,
                        num_results=5,
                        topic="general",
                    )

                    execution_log.append(
                        f"Fallback search completed with {len(website_search_results.get('results', []))} results"
                    )

                    # Format website search results for LLM
                    website_search_text = f"""
Find the official website URL for: {business_name} located in {location}

Search Results:
{website_search_results.get('answer', '')}

Details:
"""
                    for idx, result in enumerate(website_search_results.get("results", [])[:5], 1):
                        website_search_text += f"\n{idx}. {result.get('title', '')}\n{result.get('content', '')}"

                    # Extract website URL using lightweight schema
                    website_discovery = await llm_service.extract_structured(
                        prompt=website_search_text,
                        response_model=WebsiteDiscovery,
                        context="Extract only the official company website URL if clearly found. Do NOT guess or invent URLs.",
                    )

                    if website_discovery and website_discovery.website_url:
                        registry_data.official_website_url = website_discovery.website_url
                        execution_log.append(f"Website URL found in fallback search: {website_discovery.website_url}")
                        logger.info(f"[{business_name}] Website URL recovered: {website_discovery.website_url}")
                    else:
                        execution_log.append("Fallback search: No website URL could be extracted")
                        logger.warning(f"[{business_name}] Website URL still not found after fallback search")

                except Exception as fallback_error:
                    logger.warning(f"[{business_name}] Fallback website search failed: {fallback_error}")
                    execution_log.append(f"Fallback search error: {fallback_error}")

            # Determine if should continue to next node
            # Allow: active, unknown, not_found (proceed to crawler for verification)
            # Block: inactive, suspended, dissolved (definitive negative signals)
            blocked_statuses = {"inactive", "suspended", "dissolved"}
            should_continue = registry_data.registry_status.lower() not in blocked_statuses

            if not should_continue:
                execution_log.append(
                    f"Route to END: Business status is '{registry_data.registry_status}' (blocked status)"
                )
            else:
                if registry_data.registry_status.lower() == "unknown":
                    logger.info(f"[{business_name}] Registry status unknown - proceeding to website crawler for verification")
                    execution_log.append("Registry status unknown - proceeding to website crawler for verification")
                elif registry_data.registry_status.lower() == "not_found":
                    logger.info(f"[{business_name}] Registry not found - proceeding to website crawler for verification")
                    execution_log.append("Registry not found - proceeding to website crawler for verification")
                else:
                    logger.info(f"[{business_name}] Registry status '{registry_data.registry_status}' - proceeding to website crawler")
                    execution_log.append(f"Proceeding to website crawler (status: {registry_data.registry_status})")
        else:
            execution_log.append("Registry verification failed - business not found")
            should_continue = False

    except Exception as e:
        logger.error(f"[{business_name}] Node 1 error: {e}", exc_info=True)
        execution_log.append(f"Node 1 error: {e}")
        should_continue = False

    elapsed = time.time() - start_time
    logger.info(f"[{business_name}] Node 1 completed in {elapsed:.2f}s")

    return {
        "registry_data": registry_data,
        "registry_search_query": registry_search_query,
        "registry_raw_results": search_results.get("results", []),
        "registry_verification_status": registry_data.registry_status if registry_data else "not_found",
        "execution_log": execution_log,
        "node_timestamps": {**state.get("node_timestamps", {}), "discovery": elapsed},
        "should_continue": should_continue,
    }
