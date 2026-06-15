import logging
from datetime import datetime
from typing import Any, Optional, List

from celery.result import AsyncResult
from src.core.celery_app import celery_app
from src.agent.tasks import process_lead_task

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.services.db_service import DatabaseService

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

class JobAccepted(BaseModel):
    job_id: str
    status: str = "queued"
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str           # queued | started | success | failure | retry
    result: Optional[dict] = None
    error: Optional[str] = None


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
# AGENTIC ENDPOINTS
# ==========================================

@app.post("/api/v1/enrich-company", response_model=JobAccepted, status_code=202)
async def enrich_company_endpoint(payload: CompanyEnrichmentRequest) -> JobAccepted:
    """
    Non-blocking. Drops the job onto the Redis queue and returns
    a job_id immediately. The client should poll GET /api/v1/jobs/{job_id}.
    
    HTTP 202 Accepted is the correct status code here — the request
    was accepted but processing has not completed.
    """
    task = process_lead_task.delay(
        payload.business_name,
        payload.investment_thesis,
        payload.website_url,
        payload.target_personas,
    )
    logger.info("[api] Queued lead enrichment job_id=%s company=%s", task.id, payload.business_name)
    return JobAccepted(
        job_id=task.id,
        status="queued",
        message=f"Job accepted. Poll /api/v1/jobs/{task.id} for status.",
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """
    Poll endpoint for Celery task status.

    Celery states map to our API response as follows:
      PENDING  → queued   (task not yet picked up by a worker)
      STARTED  → started  (worker has begun execution)
      SUCCESS  → success  (result is available)
      FAILURE  → failure  (terminal error, check .error field)
      RETRY    → retry    (transient failure, worker will re-attempt)
    """
    result = AsyncResult(job_id, app=celery_app)
    status_map = {
        "PENDING": "queued",
        "STARTED": "started",
        "SUCCESS": "success",
        "FAILURE": "failure",
        "RETRY": "retry",
    }
    api_status = status_map.get(result.status, result.status.lower())

    task_result = None
    error_msg = None

    if result.successful():
        task_result = result.result  # the dict returned by process_lead_task
    elif result.failed():
        error_msg = str(result.result)  # result.result holds the exception on failure

    return JobStatus(
        job_id=job_id,
        status=api_status,
        result=task_result,
        error=error_msg,
    )


# ==========================================
# DATA RETRIEVAL ENDPOINTS
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
        raise HTTPException(status_code=500, detail=f"Failed to fetch leads: {e}") from e


@app.post("/api/search")
async def search_leads(query: str):
    """Placeholder for pgvector semantic search."""
    return {"message": "Vector similarity search not yet implemented", "query": query}