"""Global state definition for the LangGraph pipeline."""

import operator
from typing import TypedDict, Optional, List, Any, Dict, Annotated
from src.models.schemas import (
    RegistryVerification,
    WebsiteSignals,
    ConsensusResult,
    HunterContact,
)


class LeadState(TypedDict, total=False):
    """
    Unified state object passed through all graph nodes.
    
    Each node reads and updates specific fields while preserving others
    for downstream consumption. The `total=False` allows optional fields.
    """

    # === Input / Query Parameters ===
    query: str
    """Natural language intent (e.g., 'Find HVAC distributors in Ohio')"""

    business_name: str
    """Primary business name to investigate"""

    location: str
    """Geographic location (city, state, region)"""

    website_url: Optional[str]
    """Known or discovered business website URL"""

    investment_thesis: Optional[str]
    """Buyer/investor thesis passed to Node 2 to shape dynamic signal extraction"""

    # === Node 1: Discovery & State Registry Check ===
    registry_search_query: Optional[str]
    """Refined Tavily search query constructed from query + location"""

    registry_data: Optional[RegistryVerification]
    """Structured extraction from state licensing registries (Node 1 output)"""

    registry_raw_results: Optional[List[Dict[str, Any]]]
    """Raw Tavily search results before structured extraction"""

    registry_verification_status: Optional[str]
    """Status: 'active', 'inactive', 'not_found', 'error'"""

    # === Node 2: Web Crawler & Signal Extraction ===
    website_markdown: Optional[str]
    """Clean Markdown version of business website (from Crawl4AI)"""

    website_crawl_success: bool
    """Whether website crawl was successful"""

    website_crawl_error: Optional[str]
    """Error message if crawl failed"""

    extracted_signals: Optional[WebsiteSignals]
    """Structured LLM extraction of hidden signals from website Markdown (Node 2 output)"""

    # === Node 3: Triangulated Consensus & Scoring ===
    consensus_result: Optional[ConsensusResult]
    """Deterministic comparison of registry vs. website data (Node 3 output)"""

    name_match_confidence: Optional[float]
    """Fuzzy match score between registry and website business names (0.0-1.0)"""

    address_match_confidence: Optional[float]
    """Fuzzy match score between registry and website addresses (0.0-1.0)"""

    lead_score: Optional[int]
    """Final lead quality score (0-100) after consensus validation"""

    consensus_passed: bool
    """Whether lead passed consensus validation"""

    dropped_reason: Optional[str]
    """Reason for dropping lead if consensus_passed=False"""

    # === Node 4: Enrichment ===
    enrichment_data: Optional[List[HunterContact]]
    """Contact information from Hunter.io for verified owner/decision makers"""

    enrichment_success: bool
    """Whether Hunter.io enrichment succeeded"""

    enrichment_error: Optional[str]
    """Error message if enrichment failed"""

    # === Execution Metadata ===
    execution_log: Annotated[List[str], operator.add]
    """Chronological log of node execution steps and decisions.
    Uses operator.add reducer to append entries instead of overwriting."""

    node_timestamps: Dict[str, float]
    """Execution timestamps for each node (for performance tracking)"""

    total_tokens_used: Optional[int]
    """Cumulative token usage across all LLM calls"""

    errors_encountered: Annotated[List[str], operator.add]
    """Collected errors that did not halt execution.
    Uses operator.add reducer to append errors instead of overwriting."""

    should_continue: bool
    """Flag for conditional edge routing (set by nodes to control graph flow)"""
