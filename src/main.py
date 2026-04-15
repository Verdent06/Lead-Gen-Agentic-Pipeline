"""Main entry point for the autonomous sourcing agent pipeline."""

import asyncio
import csv
import logging
import os
import time
from typing import Optional
from src.models.state import LeadState
from src.models.schemas import FinalLeadOutput
from src.graph import build_graph
from src.config import Config
from pydantic import BaseModel, Field
from typing import List

class DiscoveredBusiness(BaseModel):
    business_name: str = Field(..., description="The name of the business")
    location: str = Field(..., description="The location (city, state) of the business")

class BusinessList(BaseModel):
    businesses: List[DiscoveredBusiness] = Field(..., description="List of discovered businesses")

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


async def discover_businesses(macro_query: str) -> List[DiscoveredBusiness]:
    """Use Tavily and LLM to find a list of businesses matching the macro-query."""
    from src.services.tavily_service import get_tavily_service
    from src.services.llm_service import get_llm_service
    
    logger.info(f"Running batch discovery for macro-query: {macro_query}")
    tavily = await get_tavily_service()
    llm = await get_llm_service()
    
    # Search Tavily
    search_results = await tavily.search(
        query=macro_query,
        include_answer=True,
        num_results=50,
        topic="general",
        search_depth="advanced"
    )
    
    # Format results for LLM
    context = "Tavily Search Results:\n"
    if search_results.get("answer"):
        context += f"Summary: {search_results['answer']}\n\n"
    
    for i, res in enumerate(search_results.get("results", [])):
        context += f"Result {i+1}:\nTitle: {res.get('title')}\nContent: {res.get('content')}\n\n"
        
    prompt = f"""Extract all real businesses mentioned in the search results that match the query: '{macro_query}'.
For each business, extract:
- Business name (exact name from the results)
- Location (city, state or country)

Return the data strictly matching the requested JSON schema.
Do not return raw arrays - return the exact JSON object structure specified in the schema."""
    
    extracted = await llm.extract_structured(
        prompt=prompt,
        response_model=BusinessList,
        context=context
    )
    
    if extracted and extracted.businesses:
        logger.info(f"Discovered {len(extracted.businesses)} businesses.")
        return extracted.businesses
    else:
        logger.warning("No businesses discovered or extraction failed.")
        return []

async def run_batch_pipeline(macro_query: str):
    """Run the pipeline concurrently for a list of discovered businesses."""
    businesses = await discover_businesses(macro_query)
    
    if not businesses:
        logger.error("No businesses found to process.")
        return []
        
    logger.info(f"Starting concurrent processing for {len(businesses)} businesses...")
    
    # Create tasks for each business
    tasks = []
    for biz in businesses:
        # Pass the macro_query as the query context for each run
        task = run_sourcing_agent(
            query=macro_query,
            business_name=biz.business_name,
            location=biz.location
        )
        tasks.append(task)
        
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Pipeline failed for {businesses[i].business_name}: {result}")
        else:
            successful_results.append(result)
            
    return successful_results


def export_qualified_leads(results: list, filename: str = "qualified_leads.csv"):
    """Export qualified leads that passed consensus to a CSV file."""
    # Filter for leads that passed consensus
    qualified = [r for r in results if r.passed_consensus]
    if not qualified:
        print("\n[EXPORT] No leads passed consensus in this batch. CSV not updated.")
        return
    
    headers = ["Business Name", "Website", "Lead Score", "Contact Name", "Contact Email", "Job Title"]
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
            
        for lead in qualified:
            c_name = f"{lead.primary_contact.first_name or ''} {lead.primary_contact.last_name or ''}".strip() if lead.primary_contact else "N/A"
            c_email = lead.primary_contact.email if lead.primary_contact else "N/A"
            c_title = lead.primary_contact.job_title if lead.primary_contact else "N/A"
            website = lead.registry_verification.official_website_url if (lead.registry_verification and lead.registry_verification.official_website_url) else "N/A"
            
            writer.writerow([lead.business_name, website, lead.lead_score, c_name, c_email, c_title])
            
    print(f"\n[EXPORT] ✅ Successfully appended {len(qualified)} highly qualified leads to {filename}")


async def main():
    """Main entry point with batch query."""
    macro_query = "List 30 independent HVAC distributors in Ohio. Do not include merged companies. List unique entities only."
    
    # Run the batch agent
    results = await run_batch_pipeline(macro_query)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"BATCH PROCESSING COMPLETE: {len(results)} businesses processed")
    print("=" * 70)
    
    for result in results:
        print(f"\nBusiness: {result.business_name}")
        print(f"Location: {result.location}")
        print(f"Lead Score: {result.lead_score}/100")
        print(f"Passed Consensus: {result.passed_consensus}")
        print(f"Recommendation: {result.recommendation}")
        print("-" * 40)
    
    export_qualified_leads(results)
        
    return results


if __name__ == "__main__":
    # Run the async pipeline
    result = asyncio.run(main())
