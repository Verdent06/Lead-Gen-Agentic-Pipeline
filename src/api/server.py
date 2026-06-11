"""FastAPI REST API for the Lead-Gen agentic pipeline."""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.services.db_service import DatabaseService

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lead-Gen Agentic Pipeline API",
    description="REST API for processed leads and semantic search",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")


class SearchResponse(BaseModel):
    message: str
    query: str
    results: list[Any] = Field(default_factory=list)


@app.on_event("startup")
async def startup() -> None:
    """Initialize the database connection pool."""
    app.state.db = DatabaseService()
    await app.state.db.init_pool()
    logger.info("API startup complete — database pool ready")


@app.on_event("shutdown")
async def shutdown() -> None:
    """Close the database connection pool."""
    db: Optional[DatabaseService] = getattr(app.state, "db", None)
    if db is not None:
        await db.close_pool()
    logger.info("API shutdown complete — database pool closed")


def _get_db() -> DatabaseService:
    db: Optional[DatabaseService] = getattr(app.state, "db", None)
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database service is not initialized",
        )
    return db


@app.get("/api/leads", response_model=LeadsResponse)
async def get_leads() -> LeadsResponse:
    """Fetch all processed leads from target_entities."""
    db = _get_db()
    try:
        rows = await db.fetch_all_leads()
        leads = [LeadRecord(**row) for row in rows]
        return LeadsResponse(leads=leads, count=len(leads))
    except Exception as e:
        logger.error("Failed to fetch leads: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch leads from database: {e}",
        ) from e


@app.post("/api/search", response_model=SearchResponse)
async def search_leads(payload: SearchRequest) -> SearchResponse:
    """Semantic search over target_entities via pgvector (placeholder)."""
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query is required and cannot be empty",
        )

    try:
        # TODO: embed query via llm_service.generate_embedding, then cosine search in pgvector
        return SearchResponse(
            message="Search endpoint placeholder — vector similarity search not yet implemented",
            query=query,
            results=[],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Search failed for query %r: %s", query, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search request failed: {e}",
        ) from e
