import asyncio
import logging
from typing import Optional, List

from celery import shared_task

from src.agent.graph import build_graph
from src.agent.state import LeadState
from src.services.db_service import DatabaseService

logger = logging.getLogger(__name__)

# Build the graph once at module import time so every worker process
# reuses the same compiled graph rather than rebuilding per task.
_graph = build_graph()


async def _run_pipeline(
    business_name: str,
    investment_thesis: str,
    website_url: Optional[str],
    target_personas: List[str],
) -> dict:
    """
    Pure async logic: invoke the LangGraph pipeline and persist the result.
    Separated from the Celery task so it's independently testable.
    """
    initial_state: LeadState = {
        "query": investment_thesis,
        "business_name": business_name,
        "location": "United States",
        "website_url": website_url,
        "industry_definition": "Determined via AI analysis",
        "investment_thesis": investment_thesis,
        "target_decision_makers": target_personas,
        "direct_to_enrichment": True,
        "execution_log": ["Celery task started"],
        "node_timestamps": {},
        "errors_encountered": [],
        "should_continue": True,
        "website_crawl_success": False,
        "enrichment_success": False,
        "consensus_passed": False,
        "enrichment_data": [],
        "primary_contact": None,
    }

    final_state = await _graph.ainvoke(initial_state)

    # Persist to database immediately after graph execution.
    # DatabaseService reads DB_HOST/DB_USER/etc. from environment,
    # which are injected by docker-compose into the worker container.
    db = DatabaseService()
    await db.init_pool()
    try:
        await db.insert_target_entity(
            url=final_state.get("website_url") or f"unknown://{business_name}",
            company_name=business_name,
            raw_content=final_state.get("raw_content") or "",
            embedding=final_state.get("embedding"),
            primary_contact=final_state.get("primary_contact"),
            all_contacts=final_state.get("enrichment_data") or [],
        )
    finally:
        await db.close_pool()

    return {
        "business_name": business_name,
        "resolved_url": final_state.get("website_url"),
        "contacts_found": len(final_state.get("enrichment_data") or []),
        "primary_contact": final_state.get("primary_contact"),
    }


@shared_task(bind=True, name="agent.tasks.process_lead_task")
def process_lead_task(
    self,
    business_name: str,
    investment_thesis: str,
    website_url: Optional[str],
    target_personas: List[str],
) -> dict:
    """
    Celery entry point. sync wrapper around the async pipeline.

    bind=True gives us `self` so we can call self.retry() on transient
    failures (network blips, rate limits) without re-queuing manually.

    asyncio.run() is the correct bridge here — it spins up a fresh event
    loop for this task and tears it down cleanly when done. Never share
    an event loop across Celery tasks.
    """
    logger.info("[worker] Processing lead: %s", business_name)
    try:
        return asyncio.run(
            _run_pipeline(business_name, investment_thesis, website_url, target_personas)
        )
    except Exception as exc:
        logger.exception("[worker] Task failed for %s: %s", business_name, exc)
        # Retry up to 3 times with 30s exponential backoff.
        # countdown doubles per attempt: 30s → 60s → 120s.
        raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries), max_retries=3)