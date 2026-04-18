"""Main entry point for the autonomous sourcing agent pipeline."""

import asyncio
import csv
import logging
import math
import os
import re
import time
from urllib.parse import urlparse
from typing import Any, Optional
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
    investment_thesis: Optional[str] = None,
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
        investment_thesis: Buyer thesis; shapes dynamic signals in Node 2

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
            "investment_thesis": investment_thesis,
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


async def discover_businesses(
    search_query: str, extraction_instructions: str, num_results_per_query: int = 20
) -> List[DiscoveredBusiness]:
    """Use Tavily fan-out + deterministic dedupe + LLM extraction to find businesses."""
    from src.services.tavily_service import get_tavily_service
    from src.services.llm_service import get_llm_service

    def _normalize_url(url: str) -> str:
        if not url:
            return ""
        raw = str(url).strip()
        if not raw:
            return ""
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = (parsed.path or "").rstrip("/")
        return f"{host}{path}"

    def _normalize_text_key(text: str) -> str:
        t = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
        return " ".join(t.split())

    def _result_key(item: dict[str, Any]) -> str:
        u = _normalize_url(item.get("url", ""))
        if u:
            return f"url::{u}"
        title = _normalize_text_key(item.get("title", ""))
        content = _normalize_text_key(item.get("content", ""))[:220]
        return f"text::{title}::{content}"

    def _biz_key(name: str, location: str) -> str:
        # Dedupe by normalized legal-ish name + location.
        n = _normalize_text_key(name)
        n = re.sub(r"\b(inc|llc|ltd|co|corp|corporation|company)\b", " ", n)
        n = " ".join(n.split())
        l = _normalize_text_key(location)
        return f"{n}::{l}"

    def _extract_target_count(instructions: str, default: int = 30) -> int:
        s = str(instructions or "").lower()
        patterns = [
            r"(?:maximum|max)\s+of\s+(\d+)",
            r"up to\s+(\d+)",
            r"extract\s+(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, s)
            if m:
                return int(m.group(1))
        return default

    logger.info(f"Running batch discovery — base Tavily search query: {search_query}")
    tavily = await get_tavily_service()
    llm = await get_llm_service()

    keyword_variants = [
        "hvac distributors",
        "hvac wholesalers",
        "hvac supply houses",
        "mechanical equipment distributors",
    ]
    city_variants = [
        "ohio",
        "cleveland ohio",
        "columbus ohio",
        "cincinnati ohio",
        "dayton ohio",
        "toledo ohio",
        "akron ohio",
    ]

    queries: list[str] = [search_query]
    for kw in keyword_variants:
        for city in city_variants:
            queries.append(f"{kw} {city}")
    # Preserve order, remove duplicates
    queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))

    requested_per_query = max(1, int(num_results_per_query))
    target_businesses = _extract_target_count(extraction_instructions, default=30)
    # Conservative priors for planning before we observe this run.
    assumed_docs_per_query = max(10, int(requested_per_query * 0.35))
    assumed_doc_unique_ratio = 0.55
    assumed_biz_per_unique_doc = 0.12
    estimated_unique_docs_needed = math.ceil(target_businesses / assumed_biz_per_unique_doc)
    estimated_raw_docs_needed = math.ceil(estimated_unique_docs_needed / assumed_doc_unique_ratio)
    planned_query_count = max(1, math.ceil(estimated_raw_docs_needed / assumed_docs_per_query))
    planned_query_count = min(planned_query_count, len(queries))
    active_queries = queries[:planned_query_count]

    logger.info(
        "Discovery planning: "
        f"target_businesses={target_businesses}, requested_per_query={requested_per_query}, "
        f"candidate_queries={len(queries)}, planned_queries={len(active_queries)}, "
        f"estimated_raw_docs_needed={estimated_raw_docs_needed}, "
        f"estimated_unique_docs_needed={estimated_unique_docs_needed}"
    )

    search_jobs = [
        tavily.search(
            query=q,
            include_answer=True,
            num_results=requested_per_query,
            topic="general",
            search_depth="basic",
        )
        for q in active_queries
    ]
    search_payloads = await asyncio.gather(*search_jobs, return_exceptions=True)

    aggregated_results: list[dict[str, Any]] = []
    summaries: list[str] = []
    per_query_return_counts: list[int] = []
    for i, payload in enumerate(search_payloads):
        if isinstance(payload, Exception):
            logger.warning(f"Discovery Tavily query failed: {active_queries[i]} ({payload})")
            per_query_return_counts.append(0)
            continue
        per_query_return_counts.append(len(payload.get("results", [])))
        ans = payload.get("answer")
        if ans:
            summaries.append(str(ans).strip())
        for item in payload.get("results", []):
            if isinstance(item, dict):
                aggregated_results.append(item)

    deduped_result_map: dict[str, dict[str, Any]] = {}
    for item in aggregated_results:
        key = _result_key(item)
        if key not in deduped_result_map:
            deduped_result_map[key] = item
            continue
        old_score = deduped_result_map[key].get("score", 0.0) or 0.0
        new_score = item.get("score", 0.0) or 0.0
        if new_score > old_score:
            deduped_result_map[key] = item

    deduped_results = list(deduped_result_map.values())
    raw_count = len(aggregated_results)
    unique_count = len(deduped_results)
    observed_avg_docs_per_query = (
        (sum(per_query_return_counts) / len(per_query_return_counts)) if per_query_return_counts else 0.0
    )
    observed_return_ratio = (
        (observed_avg_docs_per_query / requested_per_query) if requested_per_query else 0.0
    )
    observed_doc_unique_ratio = (unique_count / raw_count) if raw_count else 0.0
    logger.info(
        "Discovery Tavily aggregation: "
        f"{raw_count} raw results → {unique_count} deduped results; "
        f"avg_docs_per_query={observed_avg_docs_per_query:.1f}/{requested_per_query} "
        f"(return_ratio={observed_return_ratio:.2f}), "
        f"doc_unique_ratio={observed_doc_unique_ratio:.2f}"
    )
    if observed_avg_docs_per_query < requested_per_query:
        logger.info(
            "Tavily returned fewer documents than requested per query. "
            "Planning and yield math now uses observed return counts instead of assuming max_results is always met."
        )
    if unique_count < estimated_unique_docs_needed:
        logger.info(
            "Discovery context below estimated target coverage: "
            f"unique_docs={unique_count} vs estimated_needed={estimated_unique_docs_needed}. "
            "Consider adding more city/keyword variants."
        )

    context = "Tavily Search Results:\n"
    for i, s in enumerate(summaries[:8], 1):
        context += f"Summary {i}: {s}\n"
    if summaries:
        context += "\n"

    for i, res in enumerate(deduped_results):
        context += f"Result {i+1}:\nTitle: {res.get('title')}\nContent: {res.get('content')}\n\n"

    prompt = f"{extraction_instructions}\n\nExtract the businesses from the following context:\n\n{context}"

    extracted = await llm.extract_structured(
        prompt=prompt,
        response_model=BusinessList,
        context="",
    )
    
    if extracted and extracted.businesses:
        deduped_businesses: list[DiscoveredBusiness] = []
        seen: set[str] = set()
        for b in extracted.businesses:
            key = _biz_key(b.business_name, b.location)
            if key in seen:
                continue
            seen.add(key)
            deduped_businesses.append(b)
        observed_biz_per_unique_doc = (
            (len(deduped_businesses) / unique_count) if unique_count else 0.0
        )
        projected_unique_docs_for_target = (
            math.ceil(target_businesses / observed_biz_per_unique_doc)
            if observed_biz_per_unique_doc > 0
            else None
        )
        logger.info(
            f"Discovered {len(extracted.businesses)} businesses from LLM; "
            f"{len(deduped_businesses)} after business dedupe. "
            f"Observed biz_per_unique_doc={observed_biz_per_unique_doc:.3f}"
        )
        if projected_unique_docs_for_target:
            logger.info(
                "Discovery projection from current run: "
                f"target={target_businesses} would need ~{projected_unique_docs_for_target} unique docs "
                "at this observed extraction yield."
            )
        return deduped_businesses
    else:
        logger.warning("No businesses discovered or extraction failed.")
        return []

async def run_batch_pipeline(
    search_query: str,
    extraction_instructions: str,
    investment_thesis: str,
    num_results_per_query: int = 20,
):
    """Run the pipeline concurrently for a list of discovered businesses."""
    businesses = await discover_businesses(
        search_query, extraction_instructions, num_results_per_query=num_results_per_query
    )

    if not businesses:
        logger.error("No businesses found to process.")
        return []

    logger.info(f"Starting concurrent processing for {len(businesses)} businesses...")

    tasks = []
    for biz in businesses:
        task = run_sourcing_agent(
            query=extraction_instructions,
            business_name=biz.business_name,
            location=biz.location,
            investment_thesis=investment_thesis,
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


_QUALIFIED_LEADS_HEADERS = [
    "Business Name",
    "Website",
    "Lead Score",
    "Contact Name",
    "Contact Email",
    "Job Title",
]


def _qualified_leads_csv_has_header(path: str) -> bool:
    """True if the file exists, is non-empty, and its first row matches our header."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, newline="", encoding="utf-8") as f:
            row = next(csv.reader(f), None)
        return row == _QUALIFIED_LEADS_HEADERS
    except (OSError, StopIteration):
        return False


def _qualified_leads_csv_prepend_header_if_missing(path: str) -> None:
    """If the CSV has data but no header row, rewrite once with the standard header."""
    if _qualified_leads_csv_has_header(path):
        return
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return
    with open(path, newline="", encoding="utf-8") as f:
        existing = list(csv.reader(f))
    if not existing:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_QUALIFIED_LEADS_HEADERS)
        w.writerows(existing)


def export_qualified_leads(results: list, filename: str = "qualified_leads.csv"):
    """Export qualified leads that passed consensus to a CSV file."""
    # Filter for leads that passed consensus
    qualified = [r for r in results if r.passed_consensus]
    if not qualified:
        print("\n[EXPORT] No leads passed consensus in this batch. CSV not updated.")
        return

    _qualified_leads_csv_prepend_header_if_missing(filename)

    new_rows = []
    for lead in qualified:
        c_name = (
            f"{lead.primary_contact.first_name or ''} {lead.primary_contact.last_name or ''}".strip()
            if lead.primary_contact
            else "N/A"
        )
        c_email = lead.primary_contact.email if lead.primary_contact else "N/A"
        c_title = lead.primary_contact.job_title if lead.primary_contact else "N/A"
        website = (
            lead.registry_verification.official_website_url
            if (lead.registry_verification and lead.registry_verification.official_website_url)
            else "N/A"
        )
        new_rows.append([lead.business_name, website, lead.lead_score, c_name, c_email, c_title])

    file_empty = not os.path.isfile(filename) or os.path.getsize(filename) == 0
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if file_empty:
            writer.writerow(_QUALIFIED_LEADS_HEADERS)
        writer.writerows(new_rows)

    print(f"\n[EXPORT] ✅ Successfully appended {len(qualified)} highly qualified leads to {filename}")


async def main():
    """Main entry point with batch query."""
    search_query = "directory list of HVAC distributors, wholesalers, and supply houses in Ohio"
    extraction_instructions = (
        "Extract a maximum of 30 unique, independent HVAC distributors in Ohio. "
        "Do NOT include merged companies. List unique entities only."
    )
    num_results_per_query = 20
    investment_thesis = (
        "We are looking for high-growth B2B HVAC distributors that may have recently acquired "
        "other branches or competitors and are insignsvesting in a modern contractor-facing e-commerce "
        "or ordering experience, with optional  of ownership transition."
    )

    results = await run_batch_pipeline(
        search_query,
        extraction_instructions,
        investment_thesis,
        num_results_per_query=num_results_per_query,
    )
    
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
