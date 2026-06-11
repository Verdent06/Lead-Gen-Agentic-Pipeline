"""FastAPI REST API for the Lead-Gen agentic pipeline."""

import logging
from datetime import datetime
from typing import Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.services.db_service import DatabaseService
from src.agent.graph import build_graph
from src.agent.state import LeadState

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lead-Gen Agentic Pipeline API",
    description="REST API for executing LangGraph AI workflows and retrieving leads.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AI Graph once at startup
graph = build_graph()


# ==========================================
# SCHEMA DEFINITIONS
# ==========================================

class LeadRecord(BaseModel):
    id: int
    url: str
    company_name: Optional[str] = None
    primary_contact: Optional[dict] = None
    all_contacts: Optional[list[dict]] = None
    raw_content: Optional[str] = None
    scraped_at: Optional[datetime] = None

class LeadsResponse(BaseModel):
    leads: list[LeadRecord]
    count: int

class CompanyEnrichmentRequest(BaseModel):
    business_name: str = Field(..., description="The name of the target company")
    website_url: Optional[str] = Field(None, description="Known URL, or None to let Tavily discover it")
    investment_thesis: str = Field(..., description="The contextual goal of the campaign")
    target_personas: List[str] = Field(default=["HRIS", "HRIT", "Enterprise Applications"])


# ==========================================
# LIFECYCLE EVENTS
# ==========================================

@app.on_event("startup")
async def startup() -> None:
    """Initialize the database connection pool."""
    app.state.db = DatabaseService()
    await app.state.db.init_pool()

def _get_db() -> DatabaseService:
    return app.state.db


# ==========================================
# AGENTIC ENDPOINTS (THE ENGINE)
# ==========================================

@app.post("/api/v1/enrich-company")
async def enrich_company_endpoint(payload: CompanyEnrichmentRequest):
    """
    Executes the LangGraph pipeline for a single company dynamically.
    """
    initial_state: LeadState = {
        "query": payload.investment_thesis,
        "business_name": payload.business_name,
        "location": "United States", 
        "website_url": payload.website_url,
        "industry_definition": "Determined via AI analysis",
        "investment_thesis": payload.investment_thesis,
        "target_decision_makers": payload.target_personas,
        "direct_to_enrichment": True,
        "execution_log": ["API request received"],
        "node_timestamps": {},
        "errors_encountered": [],
        "should_continue": True,
        "website_crawl_success": False,
        "enrichment_success": False,
        "consensus_passed": False,
        "enrichment_data": [],
        "primary_contact": None
    }

    try:
        logger.info(f"[*] API triggered graph execution for: {payload.business_name}")
        final_state = await graph.ainvoke(initial_state)
        
        # Note: In a true production app, you would also call db.insert_target_entity here
        # to ensure the API execution actually saves to the database immediately.
        
        return {
            "status": "success",
            "business_name": payload.business_name,
            "resolved_url": final_state.get("website_url"),
            "contacts_found": len(final_state.get("enrichment_data", [])),
            "primary_contact": final_state.get("primary_contact"),
        }
        
    except Exception as e:
        logger.error(f"Graph execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")


# ==========================================
# DATA RETRIEVAL ENDPOINTS (CRUD)
# ==========================================

@app.get("/leads", response_model=LeadsResponse)
async def get_leads() -> LeadsResponse:
    """Fetch all processed leads from target_entities."""
    db = _get_db()
    try:
        rows = await db.fetch_all_leads()
        leads = [LeadRecord(**row) for row in rows]
        return LeadsResponse(leads=leads, count=len(leads))
    except Exception as e:
        logger.error("Failed to fetch leads: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch leads from database: {e}") from e


@app.post("/api/search")
async def search_leads(query: str):
    """Placeholder for pgvector semantic search."""
    return {"message": "Vector similarity search not yet implemented", "query": query}