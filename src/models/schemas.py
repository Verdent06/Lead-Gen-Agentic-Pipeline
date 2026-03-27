"""Pydantic schemas for strict LLM output validation."""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator, HttpUrl


class RegistryVerification(BaseModel):
    """
    Structured output from Node 1 (Discovery & Registry Check).
    Extracted from state licensing registry via LLM analysis of Tavily search results.
    """

    business_name: str = Field(..., description="Legal business name from registry")

    dba_names: Optional[List[str]] = Field(
        default=None, description="Doing Business As names if available"
    )
    
    official_website_url: Optional[str] = Field(
        default=None, description="The official company website URL if found in search results."
    )

    registry_status: Optional[str] = Field(
        default="unknown", description="Current business status in state registry"
    )

    registry_url: Optional[str] = Field(
        default=None, description="Direct link to registry record"
    )

    business_address: Optional[str] = Field(default=None, description="Physical address from registry")

    city: Optional[str] = Field(default=None)
    state: Optional[str] = Field(default=None)
    zip_code: Optional[str] = Field(default=None)

    incorporation_date: Optional[str] = Field(
        default=None, description="Date business was incorporated/registered (YYYY-MM-DD)"
    )

    owner_name: Optional[str] = Field(default=None, description="Primary owner/CEO name")

    owner_email: Optional[str] = Field(default=None, description="Owner email if publicly listed")

    business_type: Optional[str] = Field(
        default=None, description="Business type (LLC, S-Corp, C-Corp, etc.)"
    )

    industry_classification: Optional[str] = Field(
        default=None, description="SIC or NAICS code/description"
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence that registry data matches the query (0.0-1.0)",
    )

    notes: Optional[str] = Field(default=None, description="Additional context or warnings")

    @validator("registry_status", pre=True)
    def normalize_status(cls, v):
        """Normalize status to lowercase."""
        return v.lower() if isinstance(v, str) else v


class WebsiteDiscovery(BaseModel):
    """
    Lightweight schema for fallback website URL discovery (Node 1 secondary search).
    Used when initial registry search doesn't include company website.
    """

    website_url: Optional[str] = Field(
        default=None,
        description="Official company website URL. Only include if explicitly found in search results. Do NOT guess or invent URLs.",
    )

    notes: Optional[str] = Field(
        default=None, description="Any relevant notes about the URL search"
    )


class SignalCategory(BaseModel):
    """Individual extracted signal with evidence."""

    signal_name: str = Field(..., description="Name of the signal (e.g., 'has_ecommerce')")

    detected: bool = Field(..., description="Whether signal is present")

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM confidence in detection (0.0-1.0)",
    )

    evidence: str = Field(..., description="Extracted text/context supporting the signal")

    raw_location: Optional[str] = Field(
        default=None, description="Where on website this was found (e.g., 'homepage footer')"
    )


class WebsiteSignals(BaseModel):
    """
    Structured output from Node 2 (Dynamic Website Crawler).
    Extracted from website Markdown via LLM with strict validation.
    """

    website_url: str = Field(..., description="URL of analyzed website")

    website_reachable: bool = Field(
        ..., description="Whether website was successfully crawled"
    )

    has_ecommerce_store: SignalCategory = Field(
        ..., description="Does business have functional e-commerce?"
    )

    ecommerce_platform: Optional[str] = Field(
        default=None,
        description="Platform detected (Shopify, WooCommerce, custom, etc.)",
    )

    legacy_software_mentions: SignalCategory = Field(
        ..., description="Mentions of outdated/legacy tech (Flash, old ASP, etc.)"
    )

    succession_planning_signals: SignalCategory = Field(
        ...,
        description="Signs of succession planning (family members, ownership transition, etc.)",
    )

    owner_retirement_mentions: SignalCategory = Field(
        ..., description="References to owner age, retirement, or succession"
    )

    business_size_indicator: Optional[str] = Field(
        default=None, description="Small, medium, large based on website indicators"
    )

    team_size_estimate: Optional[int] = Field(
        default=None, description="Estimated employee count from website"
    )

    contact_information: Optional[Dict[str, str]] = Field(
        default=None,
        description="Extracted contact: email, phone, address if present",
    )

    social_media_links: Optional[List[str]] = Field(
        default=None, description="LinkedIn, Facebook, Twitter profiles if present"
    )

    owner_name_from_site: Optional[str] = Field(
        default=None, description="Owner/CEO name if mentioned on website"
    )

    business_name_from_site: Optional[str] = Field(
        default=None, description="Business name as stated on website (for name matching)"
    )

    owner_email_from_site: Optional[str] = Field(
        default=None, description="Owner email if publicly listed"
    )

    new_domain_redirect: Optional[str] = Field(
        default=None, description="If website states domain has moved, the new domain URL"
    )

    years_in_business: Optional[int] = Field(
        default=None, description="Business vintage/age if stated on site"
    )

    additional_signals: Optional[List[SignalCategory]] = Field(
        default=None,
        description="Other custom signals extracted (expand as needed per use case)",
    )

    is_target_industry: bool = Field(
        ..., 
        description="True if the website explicitly belongs to the target industry (e.g., HVAC distribution). False if industry is irrelevant (e.g., labor union, non-HVAC business)."
    )

    industry_evidence: str = Field(
        ...,
        description="Direct evidence from website text justifying the is_target_industry determination. Include specific phrases or indicators that confirm or reject HVAC distribution industry."
    )

    extraction_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall LLM confidence in website extraction (0.0-1.0)",
    )

    notes: Optional[str] = Field(default=None, description="Extraction notes or caveats")


class ConsensusResult(BaseModel):
    """
    Output from Node 3 (Triangulated Consensus & Scoring).
    Deterministic comparison of registry data vs. website signals.
    """

    lead_should_proceed: bool = Field(
        ...,
        description="Whether lead passes all consensus checks and should proceed to enrichment",
    )

    name_match: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fuzzy match score between registry and website business names",
    )

    address_match: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fuzzy match score between registry and website addresses",
    )

    address_mismatch_details: Optional[str] = Field(
        default=None, description="Details if address does not match"
    )

    registry_website_conflict: bool = Field(
        ...,
        description="True if registry and website data significantly conflict",
    )

    conflict_description: Optional[str] = Field(
        default=None, description="Details of conflict if registry_website_conflict=True"
    )

    signal_scoring: Dict[str, int] = Field(
        ...,
        description="Points awarded for each detected signal (e.g., {'no_ecommerce': 20, 'legacy_tech': 15})",
    )

    base_signal_score: int = Field(
        ..., ge=0, le=100, description="Score from signal presence (0-100)"
    )

    match_bonus: int = Field(
        ...,
        ge=0,
        le=20,
        description="Bonus points for high name/address match confidence (0-20)",
    )

    final_lead_score: int = Field(
        ..., ge=0, le=100, description="Final consensus lead score (0-100)"
    )

    drop_reason: Optional[str] = Field(
        default=None,
        description="If lead_should_proceed=False, reason for rejection",
    )

    validation_notes: Optional[str] = Field(
        default=None, description="Additional validation context"
    )

    recommended_follow_up: Optional[str] = Field(
        default=None,
        description="Suggested next action (e.g., 'Manual review recommended')",
    )


class HunterContact(BaseModel):
    """Individual contact from Hunter.io enrichment."""

    first_name: Optional[str] = Field(default=None)

    last_name: Optional[str] = Field(default=None)

    email: str = Field(..., description="Email address")

    email_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Hunter.io confidence score for this email",
    )

    phone: Optional[str] = Field(default=None)

    job_title: Optional[str] = Field(default=None)

    department: Optional[str] = Field(default=None)

    linkedin_profile: Optional[str] = Field(default=None)

    is_owner_or_decision_maker: bool = Field(
        default=False,
        description="Flag if this contact is likely owner or key decision maker",
    )

    source: str = Field(default="hunter.io", description="Data source (hunter.io, etc.)")


class FinalLeadOutput(BaseModel):
    """
    Complete output after all 4 nodes have executed.
    This is what gets returned to the end user.
    """

    query: str = Field(..., description="Original search query")

    business_name: str = Field(..., description="Primary business name")

    location: str = Field(..., description="Geographic location")

    lead_score: int = Field(
        ..., ge=0, le=100, description="Final lead quality score (0-100)"
    )

    passed_consensus: bool = Field(..., description="Whether lead passed consensus validation")

    registry_verification: Optional[RegistryVerification] = Field(
        default=None, description="Verified registry data"
    )

    website_signals: Optional[WebsiteSignals] = Field(
        default=None, description="Extracted website signals"
    )

    consensus_details: Optional[ConsensusResult] = Field(
        default=None, description="Consensus validation results"
    )

    enriched_contacts: Optional[List[HunterContact]] = Field(
        default=None, description="Hunter.io enriched contact information"
    )

    primary_contact: Optional[HunterContact] = Field(
        default=None,
        description="Best identified owner/decision maker contact",
    )

    execution_time_seconds: Optional[float] = Field(
        default=None, description="Total execution time"
    )

    execution_log: List[str] = Field(
        default_factory=list, description="Chronological execution log"
    )

    errors_encountered: List[str] = Field(
        default_factory=list, description="Non-fatal errors during execution"
    )

    recommendation: Optional[str] = Field(
        default=None,
        description="Final recommendation (e.g., 'High-priority prospect', 'Manual review needed', etc.)",
    )
