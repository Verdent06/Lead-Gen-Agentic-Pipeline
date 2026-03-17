"""Node 1: Discovery & State Registry Verification."""

import logging
import time
from src.models.state import LeadState
from src.models.schemas import RegistryVerification
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
    logger.info("=== Node 1: Discovery & Registry Verification ===")

    execution_log = state.get("execution_log", [])
    registry_search_query = None
    search_results = {}
    registry_data = None
    should_continue = False

    try:
        # Extract input parameters
        query = state.get("query", "")
        business_name = state.get("business_name", "")
        location = state.get("location", "")

        if not business_name or not location:
            raise ValueError("business_name and location are required")

        # Construct refined search query
        registry_search_query = f"{business_name} {location} state licensing registry active business"
        logger.info(f"Searching for: {registry_search_query}")

        # Perform Tavily search
        tavily_service = await get_tavily_service()
        search_results = await tavily_service.search(
            query=registry_search_query,
            include_answer=True,
            num_results=10,
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
            logger.info(f"Business status: {registry_data.registry_status}")

            # Determine if should continue to next node
            should_continue = registry_data.registry_status == "active"

            if not should_continue:
                execution_log.append(
                    f"Route to END: Business status is '{registry_data.registry_status}'"
                )
        else:
            execution_log.append("Registry verification failed - business not found")
            should_continue = False

    except Exception as e:
        logger.error(f"Node 1 error: {e}", exc_info=True)
        execution_log.append(f"Node 1 error: {e}")
        should_continue = False

    elapsed = time.time() - start_time
    logger.info(f"Node 1 completed in {elapsed:.2f}s")

    return {
        "registry_data": registry_data,
        "registry_search_query": registry_search_query,
        "registry_raw_results": search_results.get("results", []),
        "registry_verification_status": registry_data.registry_status if registry_data else "not_found",
        "execution_log": execution_log,
        "node_timestamps": {**state.get("node_timestamps", {}), "discovery": elapsed},
        "should_continue": should_continue,
    }
