"""Main entry point for the autonomous sourcing agent pipeline."""

import asyncio
import logging
import time
from typing import Optional
from src.models.state import LeadState
from src.models.schemas import FinalLeadOutput
from src.graph import build_graph
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_sourcing_agent(
    query: str,
    business_name: str,
    location: str,
    website_url: Optional[str] = None,
) -> FinalLeadOutput:
    """
    Execute the complete autonomous sourcing agent pipeline.

    Orchestrates all 4 nodes:
    1. Discovery & State Registry Check (Tavily)
    2. Dynamic Website Crawler & Signal Extraction (Crawl4AI + LLM)
    3. Triangulated Consensus & Scoring (Deterministic Python)
    4. Enrichment (Hunter.io)

    Args:
        query: Natural language intent (e.g., "Find HVAC distributors in Ohio")
        business_name: Business name to investigate
        location: Geographic location (city, state, region)
        website_url: Optional known website URL (otherwise derived)

    Returns:
        FinalLeadOutput with complete lead information and scoring
    """
    logger.info("=" * 70)
    logger.info("AUTONOMOUS SOURCING AGENT INITIATED")
    logger.info("=" * 70)
    logger.info(f"Query: {query}")
    logger.info(f"Business: {business_name}")
    logger.info(f"Location: {location}")

    pipeline_start = time.time()

    try:
        # Initialize state
        initial_state: LeadState = {
            "query": query,
            "business_name": business_name,
            "location": location,
            "website_url": website_url,
            "execution_log": ["Pipeline started"],
            "node_timestamps": {},
            "errors_encountered": [],
            "should_continue": True,
            "website_crawl_success": False,
            "enrichment_success": False,
            "consensus_passed": False,
        }

        # Build and execute graph
        graph = build_graph()
        logger.info("Graph built, executing with ainvoke()...")

        # Execute graph asynchronously using native ainvoke()
        final_state = await graph.ainvoke(initial_state)

        execution_time = time.time() - pipeline_start

        # Extract results from final state
        registry_data = final_state.get("registry_data")
        extracted_signals = final_state.get("extracted_signals")
        consensus_result = final_state.get("consensus_result")
        enrichment_data = final_state.get("enrichment_data", [])
        primary_contact = final_state.get("primary_contact")
        lead_score = final_state.get("lead_score", 0)
        consensus_passed = final_state.get("consensus_passed", False)
        execution_log = final_state.get("execution_log", [])
        errors = final_state.get("errors_encountered", [])

        # Determine recommendation
        if consensus_passed and lead_score >= 80:
            recommendation = "High-priority prospect - Ready for outreach"
        elif consensus_passed and lead_score >= 70:
            recommendation = "Qualified lead - Recommend follow-up"
        elif consensus_passed:
            recommendation = "Potential lead - Consider for future outreach"
        else:
            recommendation = "Lead rejected - Does not meet criteria"

        # Build final output
        final_output = FinalLeadOutput(
            query=query,
            business_name=business_name,
            location=location,
            lead_score=lead_score,
            passed_consensus=consensus_passed,
            registry_verification=registry_data,
            website_signals=extracted_signals,
            consensus_details=consensus_result,
            enriched_contacts=enrichment_data,
            primary_contact=primary_contact,
            execution_time_seconds=execution_time,
            execution_log=execution_log,
            errors_encountered=errors,
            recommendation=recommendation,
        )

        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Lead Score: {lead_score}/100")
        logger.info(f"Status: {'PASSED' if consensus_passed else 'REJECTED'}")
        logger.info(f"Execution Time: {execution_time:.2f}s")
        logger.info(f"Recommendation: {recommendation}")

        return final_output

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        execution_time = time.time() - pipeline_start

        return FinalLeadOutput(
            query=query,
            business_name=business_name,
            location=location,
            lead_score=0,
            passed_consensus=False,
            execution_time_seconds=execution_time,
            execution_log=["Pipeline failed"],
            errors_encountered=[str(e)],
            recommendation="Pipeline error - manual review required",
        )


async def main():
    """Main entry point with example query."""
    # Example query
    example_query = "Find independent HVAC distributors in Ohio that do not have e-commerce"
    example_business = "Smith HVAC Distributors"
    example_location = "Cleveland, Ohio"

    # Run the agent
    result = await run_sourcing_agent(
        query=example_query,
        business_name=example_business,
        location=example_location,
    )

    # Print results
    print("\n" + "=" * 70)
    print("FINAL LEAD OUTPUT")
    print("=" * 70)
    print(f"Business: {result.business_name}")
    print(f"Location: {result.location}")
    print(f"Lead Score: {result.lead_score}/100")
    print(f"Passed Consensus: {result.passed_consensus}")
    print(f"Recommendation: {result.recommendation}")

    if result.primary_contact:
        print(f"\nPrimary Contact:")
        print(f"  Name: {result.primary_contact.first_name} {result.primary_contact.last_name}")
        print(f"  Email: {result.primary_contact.email}")
        print(f"  Title: {result.primary_contact.job_title}")

    print(f"\nExecution Time: {result.execution_time_seconds:.2f}s")
    print(f"Errors: {len(result.errors_encountered)}")

    if result.execution_log:
        print(f"\nExecution Log ({len(result.execution_log)} entries):")
        for i, log_entry in enumerate(result.execution_log[:5], 1):
            print(f"  {i}. {log_entry}")
        if len(result.execution_log) > 5:
            print(f"  ... and {len(result.execution_log) - 5} more")

    return result


if __name__ == "__main__":
    # Run the async pipeline
    result = asyncio.run(main())
