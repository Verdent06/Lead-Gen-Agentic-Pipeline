import asyncio
from typing import List, Dict
from pydantic import BaseModel
from src.services.crawl4ai_service import get_crawl4ai_service
from src.services.llm_service import get_llm_service
from src.graph import build_graph
from src.models.state import LeadState

graph = build_graph()

# Define the schema for our LLM to strictly output the directory list
class ClientDirectory(BaseModel):
    companies: List[Dict[str, str]] # Expecting {"business_name": "...", "website_url": "..."}

async def extract_directory_clients(directory_url: str) -> List[Dict[str, str]]:
    """
    Scrapes the directory page and uses an LLM to extract all client companies.
    """
    print(f"[*] Crawling directory URL: {directory_url}")
    
    # 1. Initialize your specific crawler singleton and extract markdown
    crawler = await get_crawl4ai_service()
    markdown = await crawler.crawl_and_convert(directory_url)
    
    if not markdown:
        print("[!] Fatal Error: Crawler returned None for the directory URL.")
        return []
    
    # 2. Force the LLM to extract structured data
    prompt = f"""
    Analyze the following markdown from a workforce software client directory page.
    Extract every single client company mentioned on this page.
    For each company, provide their 'business_name' and their 'website_url' if available. 
    If the URL is not directly available, infer it (e.g., 'Acme Corp' -> 'acmecorp.com') or leave it blank to let Tavily handle it later.
    
    Markdown:
    {markdown}
    """
    
    print("[*] Extracting company nodes via LLM...")
    llm = await get_llm_service()
    response = await llm.extract_structured(prompt, response_model=ClientDirectory)
    if not response or not response.companies:
        return []
    return response.companies

async def process_single_company(semaphore: asyncio.Semaphore, company: Dict[str, str], thesis: str, personas: List[str]):
    """
    Runs the existing LangGraph pipeline for a single company, gated by a concurrency semaphore.
    """
    async with semaphore:
        print(f"[-] Initializing graph for: {company.get('business_name')}")
        
        # Initialize your existing state, but dynamically injected
        initial_state: LeadState = {
            "query": thesis,
            "business_name": company.get("business_name"),
            "location": company.get("location", "United States"),
            "website_url": company.get("website_url"),
            "industry_definition": (
                "Look for evidence that they use workforce software "
                "(e.g., Workday, SAP, UKG, Dayforce) in their careers page or software stack."
            ),
            "investment_thesis": thesis,
            "target_decision_makers": personas,
            "direct_to_enrichment": True,
            "execution_log": ["Orchestrator campaign started"],
            "node_timestamps": {},
            "errors_encountered": [],
            "should_continue": True,
            "website_crawl_success": False,
            "enrichment_success": False,
            "consensus_passed": False,
        }

        try:
            result = await graph.ainvoke(initial_state)
            print(f"[+] Successfully processed: {company.get('business_name')} - Found {len(result.get('enrichment_data', []))} contacts")
            return result
        except Exception as e:
            print(f"[!] Graph execution failed for {company.get('business_name')}: {str(e)}")
            return None

async def run_campaign():
    # Configuration for the workforce consultant client
    DIRECTORY_URL = "https://workforcesoftware.com/customers/" # REPLACE WITH ACTUAL URL
    INVESTMENT_THESIS = "Looking for companies that have recently deployed workforce software and might need post-go-live managed services."
    TARGET_PERSONAS = ["HR", "Human Resources", "Talent Acquisition", "CHRO", "Director of People"]
    
    # Throttle to 5 concurrent runs so you don't hit OpenAI/Hunter.io rate limits
    CONCURRENCY_LIMIT = 5 
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # Step 1: Map (Extract targets)
    target_companies = await extract_directory_clients(DIRECTORY_URL)
    print(f"[*] Successfully extracted {len(target_companies)} target companies from directory.")

    # Step 2 & 3: Scatter-Gather (Fan-out graph executions)
    tasks = [
        process_single_company(semaphore, company, INVESTMENT_THESIS, TARGET_PERSONAS)
        for company in target_companies
    ]
    
    print(f"[*] Launching {len(tasks)} parallel LangGraph executions with max concurrency of {CONCURRENCY_LIMIT}...")
    results = await asyncio.gather(*tasks)
    
    # Compile final results for the client
    successful_runs = [res for res in results if res is not None]
    print(f"[*] Campaign Complete. Successfully enriched {len(successful_runs)} out of {len(target_companies)} companies.")

if __name__ == "__main__":
    asyncio.run(run_campaign())