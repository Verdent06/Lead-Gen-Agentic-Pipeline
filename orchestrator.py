import asyncio
import json
from typing import List, Dict
from pydantic import BaseModel
from src.services.crawl4ai_service import crawl_website  # Assuming this exists based on your stack
from src.services.llm_service import get_llm_response    # Assuming this exists based on your stack
from src.graph import app  # Your compiled LangGraph instance
from src.models.state import LeadState

# Define the schema for our LLM to strictly output the directory list
class ClientDirectory(BaseModel):
    companies: List[Dict[str, str]] # Expecting {"business_name": "...", "website_url": "..."}

async def extract_directory_clients(directory_url: str) -> List[Dict[str, str]]:
    """
    Scrapes the directory page and uses an LLM to extract all 69+ client companies.
    """
    print(f"[*] Crawling directory URL: {directory_url}")
    # 1. Scrape the raw directory page
    crawl_result = await crawl_website(directory_url)
    
    # 2. Force the LLM to extract structured data
    prompt = f"""
    Analyze the following markdown from a workforce software client directory page.
    Extract every single client company mentioned on this page.
    For each company, provide their 'business_name' and their 'website_url' if available. 
    If the URL is not directly available, infer it (e.g., 'Acme Corp' -> 'acmecorp.com') or leave it blank to let Tavily handle it later.
    
    Markdown:
    {crawl_result.markdown}
    """
    
    print("[*] Extracting company nodes via LLM...")
    response = await get_llm_response(prompt, response_model=ClientDirectory)
    return response.companies

async def process_single_company(semaphore: asyncio.Semaphore, company: Dict[str, str], thesis: str, personas: List[str]):
    """
    Runs the existing LangGraph pipeline for a single company, gated by a concurrency semaphore.
    """
    async with semaphore:
        print(f"[-] Initializing graph for: {company.get('business_name')}")
        
        # Initialize your existing state, but dynamically injected
        initial_state = {
            "business_name": company.get("business_name"),
            "website_url": company.get("website_url"),
            "industry_definition": "Look for evidence that they use workforce software (e.g., Workday, SAP, UKG, Dayforce) in their careers page or software stack.",
            "investment_thesis": thesis,
            "target_decision_makers": personas,
            "should_continue": True # Bypass initial Tavily discovery if we already have the URL
        }
        
        try:
            # Execute your LangGraph pipeline asynchronously
            result = await app.ainvoke(initial_state)
            print(f"[+] Successfully processed: {company.get('business_name')} - Found {len(result.get('enriched_contacts', []))} contacts")
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