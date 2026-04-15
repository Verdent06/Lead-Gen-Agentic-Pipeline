"""Node 3: Triangulated Consensus & Deterministic Scoring (PURE PYTHON, NO LLM)."""

import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set

from src.models.state import LeadState
from src.models.schemas import ConsensusResult

logger = logging.getLogger(__name__)

# Scoring configuration
SIGNAL_SCORES = {
    "no_ecommerce": 25,  # Positive signal: no modern e-commerce
    "legacy_software": 20,  # Positive signal: legacy tech = manual processes
    "succession_planning": 20,  # Positive signal: potential ownership transition
    "owner_retirement": 25,  # Positive signal: owner considering retirement
}

MIN_NAME_MATCH = 0.70  # Minimum fuzzy match for business names
MIN_ADDRESS_MATCH = 0.65  # Minimum fuzzy match for addresses (more flexible due to formatting)
CONSENSUS_THRESHOLD = 70  # Final score must be >= this to pass

# Truncate long blobs in logs (contact JSON can be huge)
_LOG_ADDR_FIELD_MAX = 280
_LOG_BLOB_PREVIEW_MAX = 500


def _truncate_for_log(s: Optional[str], max_len: int = _LOG_ADDR_FIELD_MAX) -> str:
    if not s or not str(s).strip():
        return "(empty)"
    t = " ".join(str(s).split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


# US state / territory full name → USPS-style abbreviation (lowercase) for address alignment
_US_STATE_NAME_TO_ABBR: Dict[str, str] = {
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "district of columbia": "dc",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "west virginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy",
}

_US_STATE_ABBRS: Set[str] = set(_US_STATE_NAME_TO_ABBR.values())

# North American phone numbers (strip before ZIP / address tokenization to avoid GIGO).
# Leading NANP country "1" must not match when it is the last digit of another number
# (e.g. ZIP 45241 before area code 513 → would otherwise strip "1 513..." and corrupt ZIP).
_NA_PHONE_STRIP_RE = re.compile(
    r"(?:"
    r"(?:\+1[-.\s]{0,3}|(?<![0-9])1[-.\s]{0,3})?"
    r"(?:\(\s*\d{3}\s*\)|\b\d{3})\s*[-.\s]?\s*\d{3}\s*[-.\s]?\s*\d{4}\b"
    r"|"
    r"(?<![0-9])\b1\d{10}\b"
    r"|"
    r"\b\d{10}\b"
    r")"
)


def _strip_north_american_phones(text: str) -> str:
    """Remove common NANP phone formats; collapse whitespace."""
    if not text:
        return ""
    s = _NA_PHONE_STRIP_RE.sub(" ", str(text))
    return re.sub(r"\s+", " ", s).strip()


_PLACEHOLDER_EXACT: Set[str] = {
    "n/a",
    "none",
    "null",
    "missing",
    "unknown",
}


def _compose_registry_address_line(registry_data: Any) -> str:
    parts = [
        getattr(registry_data, "business_address", None) or "",
        getattr(registry_data, "city", None) or "",
        getattr(registry_data, "state", None) or "",
        getattr(registry_data, "zip_code", None) or "",
    ]
    return " ".join(p.strip() for p in parts if p and str(p).strip()).strip()


def _compose_website_address_line(contact_information: Optional[Dict[str, str]]) -> str:
    """Merge common LLM keys into one string for comparison."""
    if not contact_information:
        return ""
    ci = {
        str(k).lower(): (str(v) if v is not None else "").strip()
        for k, v in contact_information.items()
        if v is not None and str(v).strip()
    }
    for key in (
        "address",
        "mailing_address",
        "hq_address",
        "physical_address",
        "street_address",
        "location",
        "street",
    ):
        if key in ci and ci[key]:
            return ci[key]
    city = ci.get("city", "")
    state = ci.get("state", "")
    z = ci.get("zip") or ci.get("postal_code") or ci.get("zipcode") or ""
    merged = " ".join(x for x in (city, state, z) if x).strip()
    return merged if len(merged) > 4 else ""


def _join_contact_values(contact_information: Optional[Dict[str, str]]) -> str:
    """All non-empty contact strings (for ZIP extraction when address key is sparse)."""
    if not contact_information:
        return ""
    parts: List[str] = []
    for v in contact_information.values():
        if v is not None and str(v).strip():
            parts.append(str(v).strip())
    return " ".join(parts)


def _extract_zip_codes(text: str) -> List[str]:
    """
    US ZIP5 or ZIP+4 → unique 5-digit codes in order of first appearance.

    Strips North American phone numbers first. A 5-digit token at the very
    start of the string (leading building number, e.g. ``11413 Enterprise …``)
    is not treated as a ZIP.
    """
    if not text:
        return []
    cleaned = _strip_north_american_phones(str(text))
    if not cleaned:
        return []
    first_tok = cleaned.split()[0]
    out: List[str] = []
    seen: Set[str] = set()
    for m in re.finditer(r"\b(\d{5})(?:-\d{4})?\b", cleaned):
        if m.start() == 0 and m.group(0) == first_tok:
            continue
        z = m.group(1)
        if z not in seen:
            seen.add(z)
            out.append(z)
    return out


def _zip_sets_from_texts(*chunks: str) -> Set[str]:
    z: Set[str] = set()
    for c in chunks:
        z.update(_extract_zip_codes(c))
    return z


@dataclass
class AddressMatchDiagnostics:
    """Structured fields logged for debugging address/ZIP/street matching."""

    registry_line_raw: str
    website_composed_raw: str
    website_contact_blob_preview: str
    registry_zip_codes: List[str]
    website_zip_codes: List[str]
    site_line_used_for_match: str
    decision_path: str
    normalized_registry: str
    normalized_website: str
    fuzzy_full_address_score: Optional[float]
    street_core_score: Optional[float]
    registry_street_candidates: List[str] = field(default_factory=list)
    website_street_candidates: List[str] = field(default_factory=list)
    overlapping_zips: List[str] = field(default_factory=list)
    zip_mismatch_detail: Optional[str] = None
    notes: str = ""


def _log_address_match_transparency(business_name: str, d: AddressMatchDiagnostics, final_score: float) -> None:
    """Emit multi-line INFO logs showing exactly what was compared."""
    logger.info(f"[{business_name}] --- Address match (transparent) ---")
    logger.info(f"[{business_name}]   Decision path: {d.decision_path}")
    if d.notes:
        logger.info(f"[{business_name}]   Notes: {d.notes}")
    logger.info(
        f"[{business_name}]   Registry address line (composed): {_truncate_for_log(d.registry_line_raw, 400)}"
    )
    logger.info(
        f"[{business_name}]   Website 'address' field (composed): {_truncate_for_log(d.website_composed_raw, 400)}"
    )
    logger.info(
        f"[{business_name}]   Website contact blob preview (ZIP mining): {_truncate_for_log(d.website_contact_blob_preview, _LOG_BLOB_PREVIEW_MAX)}"
    )
    logger.info(
        f"[{business_name}]   ZIP codes — registry: {d.registry_zip_codes if d.registry_zip_codes else '(none parsed)'} | "
        f"website (composed+blob): {d.website_zip_codes if d.website_zip_codes else '(none parsed)'}"
    )
    if d.overlapping_zips:
        logger.info(f"[{business_name}]   ZIP overlap used for match: {d.overlapping_zips}")
    if d.zip_mismatch_detail:
        logger.info(f"[{business_name}]   ZIP conflict: {d.zip_mismatch_detail}")
    logger.info(
        f"[{business_name}]   Site line used for fuzzy/street: {_truncate_for_log(d.site_line_used_for_match, 400)}"
    )
    logger.info(
        f"[{business_name}]   Normalized registry: {_truncate_for_log(d.normalized_registry, 400)}"
    )
    logger.info(
        f"[{business_name}]   Normalized website (same basis): {_truncate_for_log(d.normalized_website, 400)}"
    )
    logger.info(
        f"[{business_name}]   Street-core candidates — registry: {d.registry_street_candidates or ['(none)']} | "
        f"website: {d.website_street_candidates or ['(none)']}"
    )
    fs = f"{d.fuzzy_full_address_score:.3f}" if d.fuzzy_full_address_score is not None else "n/a"
    ss = f"{d.street_core_score:.3f}" if d.street_core_score is not None else "n/a"
    logger.info(
        f"[{business_name}]   Subscores — full-address fuzzy: {fs} | street-core: {ss} → "
        f"final address score: {final_score:.3f}"
    )


def zip_focused_address_match_score(
    registry_line: str,
    website_composed_line: str,
    website_contact_blob: str,
) -> tuple[float, Optional[str], AddressMatchDiagnostics]:
    """
    Prefer matching on US ZIP codes (registry vs site contact text).
    If both sides expose at least one ZIP and none overlap → strong mismatch.
    Otherwise fall back to max(full-address fuzzy, street-core fuzzy) on the
    composed website line (and registry line).

    Returns:
        (score 0..1, optional ZIP conflict detail, diagnostics for logging).
    """
    site_line = (website_composed_line or website_contact_blob or "").strip()
    blob_preview = _truncate_for_log(website_contact_blob, _LOG_BLOB_PREVIEW_MAX)

    reg_zip_list = _extract_zip_codes(registry_line)
    web_zip_list = _extract_zip_codes(
        f"{website_composed_line} {website_contact_blob}".strip()
    )
    reg_zips = _zip_sets_from_texts(registry_line)
    web_zips = _zip_sets_from_texts(website_composed_line, website_contact_blob)

    norm_reg = normalize_address_for_consensus(registry_line)
    norm_web = normalize_address_for_consensus(site_line) if site_line else ""

    reg_street = _street_core_candidates(norm_reg) if norm_reg else []
    web_street = _street_core_candidates(norm_web) if norm_web else []

    fuzzy = address_similarity_score(registry_line, site_line) if site_line else 0.0
    street = street_similarity_score(registry_line, site_line) if site_line else 0.0

    def _base_diag(
        path: str,
        overlap: Optional[List[str]] = None,
        zdetail: Optional[str] = None,
        notes: str = "",
    ) -> AddressMatchDiagnostics:
        return AddressMatchDiagnostics(
            registry_line_raw=registry_line or "",
            website_composed_raw=website_composed_line or "",
            website_contact_blob_preview=blob_preview,
            registry_zip_codes=reg_zip_list,
            website_zip_codes=web_zip_list,
            site_line_used_for_match=site_line,
            decision_path=path,
            normalized_registry=norm_reg,
            normalized_website=norm_web,
            fuzzy_full_address_score=fuzzy if site_line else None,
            street_core_score=street if site_line else None,
            registry_street_candidates=reg_street,
            website_street_candidates=web_street,
            overlapping_zips=list(overlap) if overlap else [],
            zip_mismatch_detail=zdetail,
            notes=notes,
        )

    if reg_zips and web_zips:
        overlap = sorted(reg_zips & web_zips)
        if overlap:
            diag = _base_diag(
                "zip_overlap_both_sides",
                overlap=overlap,
                notes="Score=1.0 from at least one matching 5-digit ZIP on registry vs website text.",
            )
            return 1.0, None, diag
        zdetail = (
            f"registry {sorted(reg_zips)} vs website {sorted(web_zips)} (no overlap)"
        )
        diag = _base_diag("zip_disjoint_both_sides", zdetail=zdetail)
        return 0.12, zdetail, diag

    if not site_line:
        diag = _base_diag(
            "no_website_location_line",
            notes="No composed address and no contact text; default score 0.5.",
        )
        diag.fuzzy_full_address_score = None
        diag.street_core_score = None
        return 0.5, None, diag

    combined = max(fuzzy, street)
    path = "fallback_max_full_fuzzy_and_street"
    if not reg_zips and web_zips:
        path = "fallback_registry_missing_zip_website_has_zip"
    elif reg_zips and not web_zips:
        path = "fallback_website_missing_zip_registry_has_zip"
    elif not reg_zips and not web_zips:
        path = "fallback_no_zip_either_side_fuzzy_plus_street"

    diag = _base_diag(
        path,
        notes="Final score = max(full-address fuzzy, street-core fuzzy).",
    )
    return combined, None, diag


def normalize_address_for_consensus(raw: str) -> str:
    """
    Deterministic normalization so registry and markdown-sourced addresses
    score fairly (punctuation, USPS tokens, state names, ZIP+4).
    """
    if not raw:
        return ""
    t = str(raw).lower()
    collapsed = re.sub(r"\s+", " ", t).strip()
    if "not listed" in collapsed or collapsed in _PLACEHOLDER_EXACT:
        return ""
    t = _strip_north_american_phones(t)
    if not t:
        return ""
    t = re.sub(r"\bp\.?\s*o\.?\s*box\s*#?\s*\w+", " pobox ", t)
    t = re.sub(r"\b(ste|suite|unit|apt|apartment)\b\.?\s*#?\s*[\w-]+", " ", t, flags=re.I)
    t = re.sub(r"[|_;]", " ", t)
    t = re.sub(r",\s*", " ", t)
    t = re.sub(r"\.\s+", " ", t)
    t = re.sub(r"\s+\.\s*", " ", t)
    for full, abbr in _US_STATE_NAME_TO_ABBR.items():
        t = re.sub(rf"\b{re.escape(full)}\b", f" {abbr} ", t)
    t = re.sub(r"\bsaint\b", " st ", t)
    _suffix_pairs = (
        (r"\bstreet\b", " st "),
        (r"\bst\b", " st "),
        (r"\bavenue\b", " ave "),
        (r"\bave\b", " ave "),
        (r"\bdrive\b", " dr "),
        (r"\bdr\b", " dr "),
        (r"\broad\b", " rd "),
        (r"\brd\b", " rd "),
        (r"\blane\b", " ln "),
        (r"\bln\b", " ln "),
        (r"\bcourt\b", " ct "),
        (r"\bct\b", " ct "),
        (r"\bboulevard\b", " blvd "),
        (r"\bblvd\b", " blvd "),
        (r"\bhighway\b", " hwy "),
        (r"\bhwy\b", " hwy "),
        (r"\broute\b", " rte "),
        (r"\bplace\b", " pl "),
        (r"\bpl\b", " pl "),
        (r"\bparkway\b", " pkwy "),
        (r"\bcircle\b", " cir "),
        (r"\bcir\b", " cir "),
        (r"\btrail\b", " trl "),
        (r"\bnortheast\b", " ne "),
        (r"\bnorthwest\b", " nw "),
        (r"\bsoutheast\b", " se "),
        (r"\bsouthwest\b", " sw "),
        (r"\bnorth\b", " n "),
        (r"\bsouth\b", " s "),
        (r"\beast\b", " e "),
        (r"\bwest\b", " w "),
    )
    for pat, rep in _suffix_pairs:
        t = re.sub(pat, rep, t, flags=re.I)
    t = re.sub(r"\b(\d{5})-\d{4}\b", r"\1", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def address_similarity_score(registry_line: str, website_line: str) -> float:
    """
    Fuzzy match on normalized addresses; also compares token-sorted forms so
    '123 main st cleveland oh 44114' aligns with 'cleveland, OH 44114 — 123 Main Street'.
    """
    a = normalize_address_for_consensus(registry_line)
    b = normalize_address_for_consensus(website_line)
    if not a or not b:
        return 0.0
    direct = fuzzy_match(a, b)
    sorted_a = " ".join(sorted(a.split()))
    sorted_b = " ".join(sorted(b.split()))
    shuffled = fuzzy_match(sorted_a, sorted_b)
    return max(direct, shuffled)


def fuzzy_match(str1: str, str2: str) -> float:
    """
    Calculate fuzzy string matching similarity (0.0-1.0).

    Uses SequenceMatcher for deterministic matching.
    """
    if not str1 or not str2:
        return 0.0

    # Normalize strings
    s1 = str(str1).lower().strip()
    s2 = str(str2).lower().strip()

    if s1 == s2:
        return 1.0

    matcher = SequenceMatcher(None, s1, s2)
    return matcher.ratio()


def _prep_address_tokens(norm: str) -> List[str]:
    """Drop trailing ZIP5 and state abbreviations from normalized address tokens."""
    toks = norm.split()
    while toks and re.fullmatch(r"\d{5}", toks[-1]):
        toks.pop()
    while toks and toks[-1] in _US_STATE_ABBRS:
        toks.pop()
    return toks


def _extract_street_core(norm_addr: str) -> str:
    """
    Street core: first token containing a digit, plus the next four tokens.
    If ``norm_addr`` has no letters (digits/spaces/punctuation only), returns "".
    """
    if not norm_addr or not str(norm_addr).strip():
        return ""
    s = str(norm_addr).strip()
    if not re.search(r"[a-z]", s, flags=re.I):
        return ""
    toks = _prep_address_tokens(s)
    if not toks:
        return ""
    start: Optional[int] = None
    for i, tok in enumerate(toks):
        if any(ch.isdigit() for ch in tok):
            start = i
            break
    if start is None:
        return ""
    chunk = toks[start : start + 5]
    return " ".join(chunk).strip()


def _street_core_candidates(norm: str) -> List[str]:
    """Normalized street-line candidate from ``_extract_street_core``."""
    core = _extract_street_core(norm)
    return [core] if core else []


def street_similarity_score(registry_line: str, website_line: str) -> float:
    """
    Compare extracted street cores (number + street name + type) with fuzzy and
    token-sorted variants so city/ZIP ordering does not dominate the score.
    """
    ac = _street_core_candidates(normalize_address_for_consensus(registry_line))
    bc = _street_core_candidates(normalize_address_for_consensus(website_line))
    if not ac or not bc:
        return 0.0
    best = 0.0
    for a in ac:
        for b in bc:
            sa = " ".join(sorted(a.split()))
            sb = " ".join(sorted(b.split()))
            best = max(best, fuzzy_match(a, b), fuzzy_match(sa, sb))
    return best


async def consensus_node(state: LeadState) -> dict:
    """
    Node 3: Triangulated Consensus & Deterministic Scoring.

    **PURE DETERMINISTIC PYTHON - NO LLM CALLS.**

    Compares registry data (Node 1) against website signals (Node 2).
    Validates:
    1. Business name fuzzy match (must be >= MIN_NAME_MATCH)
    2. Address / ZIP match (ZIP overlap when both sides have ZIPs; else fuzzy
       address similarity; must be >= MIN_ADDRESS_MATCH when site has address/contact text)
    3. No conflicting data between registry and website

    Calculates final_lead_score (0-100) based on:
    - Signal presence from Node 2
    - Match quality bonuses
    - Registry status validation

    If validation fails, drops lead with reason.

    Args:
        state: Current LeadState from graph

    Returns:
        Updated state dict with consensus result and final score
    """
    start_time = time.time()
    
    # Extract business name for logging
    business_name = state.get("business_name", "Unknown")
    logger.info(f"\n----------------------------------------\n[{business_name}] === Node 3: Triangulated Consensus & Deterministic Scoring ===")

    execution_log = state.get("execution_log", [])
    lead_should_proceed = False
    consensus_result = None
    final_lead_score = 0

    try:
        registry_data = state.get("registry_data")
        extracted_signals = state.get("extracted_signals")

        # Validate we have required data from both nodes
        if not registry_data:
            raise ValueError("No registry data from Node 1")
        if not extracted_signals:
            raise ValueError("No website signals from Node 2")

        execution_log.append("Consensus validation started")

        # === REGISTRY STATUS CHECK: Hard Fail for Inactive Statuses ===
        registry_status = (registry_data.registry_status or "unknown").lower()
        blocked_statuses = {"inactive", "suspended", "dissolved"}
        
        if registry_status in blocked_statuses:
            logger.warning(f"[{business_name}] HARD FAIL: Business legally {registry_status}")
            execution_log.append(f"REJECTED: Business legally {registry_status}")
            
            consensus_result = ConsensusResult(
                lead_should_proceed=False,
                name_match=0.0,
                address_match=0.0,
                address_mismatch_details=None,
                registry_website_conflict=True,
                conflict_description=f"Business legally {registry_status}",
                signal_scoring={},
                base_signal_score=0,
                match_bonus=0,
                final_lead_score=0,
                drop_reason="Business legally inactive",
                validation_notes=f"Registry status: {registry_status}",
                recommended_follow_up="Not a valid business - skip.",
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[{business_name}] Node 3 completed in {elapsed:.2f}s (legal status rejection)")
            
            return {
                "consensus_result": consensus_result,
                "name_match_confidence": 0.0,
                "address_match_confidence": 0.0,
                "lead_score": 0,
                "consensus_passed": False,
                "dropped_reason": "Business legally inactive",
                "execution_log": execution_log,
                "node_timestamps": {**state.get("node_timestamps", {}), "consensus": elapsed},
                "should_continue": False,
            }

        # === REGISTRY BONUS: Verification Premium ===
        registry_bonus = 0
        if registry_status == "active":
            registry_bonus = 15
            logger.info(f"[{business_name}] Registry Verification Bonus: +{registry_bonus} pts (business verified as active)")
            execution_log.append(f"+ {registry_bonus} pts: Registry verification bonus (business verified as active)")
        elif registry_status == "unknown":
            logger.info(f"[{business_name}] Registry status: unknown (no bonus, requires website verification)")
            execution_log.append("Registry status: unknown (no bonus applied, requires website verification)")

        # === STEP 1: Name Matching ===
        registry_name = registry_data.business_name or ""
        website_name = extracted_signals.business_name_from_site or registry_data.business_name or ""

        name_match = fuzzy_match(registry_name, website_name)
        logger.info(f"[{business_name}] Name match: {registry_name} vs {website_name} = {name_match:.2f}")
        execution_log.append(f"Name match score: {name_match:.2f}")

        # === STEP 2: Address Matching (normalized registry vs normalized website text) ===
        registry_address_raw = _compose_registry_address_line(registry_data)
        website_address_raw = _compose_website_address_line(
            extracted_signals.contact_information
        )
        website_contact_blob = _join_contact_values(
            extracted_signals.contact_information
        )
        has_site_location_text = bool(
            (website_address_raw or "").strip()
            or (website_contact_blob or "").strip()
        )

        address_zip_mismatch_detail: Optional[str] = None
        if has_site_location_text:
            (
                address_match,
                address_zip_mismatch_detail,
                addr_diag,
            ) = zip_focused_address_match_score(
                registry_address_raw,
                website_address_raw,
                website_contact_blob,
            )
            _log_address_match_transparency(business_name, addr_diag, address_match)
        else:
            address_match = 0.5
            logger.info(
                f"[{business_name}] --- Address match (transparent) ---\n"
                f"  No website location text (empty composed address and empty contact values). "
                f"Default address score: {address_match:.2f}"
            )
            execution_log.append(
                f"Address: no site text — default score {address_match:.2f}"
            )

        if has_site_location_text:
            execution_log.append(
                f"Address match {address_match:.2f} (path={addr_diag.decision_path}; "
                f"registry ZIPs={addr_diag.registry_zip_codes or '[]'}; "
                f"website ZIPs={addr_diag.website_zip_codes or '[]'})"
            )

        logger.info(f"[{business_name}] Address match score (summary): {address_match:.2f}")

        # === STEP 3: HARD FAIL - Industry Verification ===
        # If the website is NOT in the target industry (HVAC distribution), immediately fail
        if not extracted_signals.is_target_industry:
            logger.warning(f"[{business_name}] HARD FAIL: Irrelevant industry. Evidence: {extracted_signals.industry_evidence}")
            execution_log.append(f"REJECTED: Irrelevant Industry - {extracted_signals.industry_evidence}")
            
            consensus_result = ConsensusResult(
                lead_should_proceed=False,
                name_match=name_match,
                address_match=address_match,
                address_mismatch_details=None,
                registry_website_conflict=True,
                conflict_description="Irrelevant Industry",
                signal_scoring={},
                base_signal_score=0,
                match_bonus=0,
                final_lead_score=0,
                drop_reason="Irrelevant Industry",
                validation_notes=f"Industry rejection: {extracted_signals.industry_evidence}",
                recommended_follow_up="Not a target industry - skip.",
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[{business_name}] Node 3 completed in {elapsed:.2f}s (industry rejection)")
            
            return {
                "consensus_result": consensus_result,
                "name_match_confidence": 0.0,
                "address_match_confidence": 0.0,
                "lead_score": 0,
                "consensus_passed": False,
                "dropped_reason": "Irrelevant Industry",
                "execution_log": execution_log,
                "node_timestamps": {**state.get("node_timestamps", {}), "consensus": elapsed},
                "should_continue": False,
            }

        # === STEP 4: Conflict Detection ===
        registry_website_conflict = False
        conflict_description = None

        # Check for critical mismatches
        if name_match < MIN_NAME_MATCH:
            registry_website_conflict = True
            conflict_description = f"Business name mismatch (similarity: {name_match:.2f} < {MIN_NAME_MATCH})"
            logger.warning(f"[{business_name}] {conflict_description}")

        if address_match < MIN_ADDRESS_MATCH and has_site_location_text:
            registry_website_conflict = True
            if address_zip_mismatch_detail:
                conflict_description = (
                    f"ZIP / address mismatch ({address_zip_mismatch_detail}; "
                    f"score {address_match:.2f} < {MIN_ADDRESS_MATCH})"
                )
            else:
                conflict_description = (
                    f"Address mismatch (similarity: {address_match:.2f} < {MIN_ADDRESS_MATCH})"
                )
            logger.warning(f"[{business_name}] {conflict_description}")

        # === STEP 5: Calculate Signal Score ===
        signal_score = 0
        signal_scoring = {}

        # Award points for positive signals
        if not extracted_signals.has_ecommerce_store.detected:
            points = SIGNAL_SCORES["no_ecommerce"]
            signal_score += points
            signal_scoring["no_ecommerce"] = points
            execution_log.append(f"+ {points} pts: No e-commerce store detected")

        if extracted_signals.legacy_software_mentions.detected:
            points = SIGNAL_SCORES["legacy_software"]
            signal_score += points
            signal_scoring["legacy_software"] = points
            execution_log.append(
                f"+ {points} pts: Legacy software mentioned (evidence: {extracted_signals.legacy_software_mentions.evidence[:50]}...)"
            )

        if extracted_signals.succession_planning_signals.detected:
            points = SIGNAL_SCORES["succession_planning"]
            signal_score += points
            signal_scoring["succession_planning"] = points
            execution_log.append(
                f"+ {points} pts: Succession planning signals detected"
            )

        if extracted_signals.owner_retirement_mentions.detected:
            points = SIGNAL_SCORES["owner_retirement"]
            signal_score += points
            signal_scoring["owner_retirement"] = points
            execution_log.append(
                f"+ {points} pts: Owner retirement mentions detected"
            )

        # Cap base signal score at 100
        base_signal_score = min(signal_score, 100)

        # === STEP 6: Match Quality Bonus ===
        match_bonus = 0
        if name_match >= 0.95 and address_match >= 0.90:
            match_bonus = 20
            execution_log.append("+ 20 pts: Excellent name/address match bonus")
        elif name_match >= 0.85 and address_match >= 0.75:
            match_bonus = 10
            execution_log.append("+ 10 pts: Good name/address match bonus")

        # === STEP 7: Final Score Calculation ===
        final_lead_score = min(base_signal_score + match_bonus + registry_bonus, 100)
        
        # Log the score breakdown
        logger.info(f"[{business_name}] Score Breakdown: base={base_signal_score} + match_bonus={match_bonus} + registry_bonus={registry_bonus} = {final_lead_score}")
        execution_log.append(f"Final Score: {base_signal_score} (base) + {match_bonus} (match) + {registry_bonus} (registry) = {final_lead_score}")

        # === STEP 8: Pass/Fail Determination ===
        if registry_website_conflict:
            lead_should_proceed = False
            drop_reason = conflict_description
            execution_log.append(f"REJECTED: {drop_reason}")
        elif final_lead_score < CONSENSUS_THRESHOLD:
            lead_should_proceed = False
            drop_reason = f"Score {final_lead_score} below threshold {CONSENSUS_THRESHOLD}"
            execution_log.append(f"REJECTED: {drop_reason}")
        else:
            lead_should_proceed = True
            drop_reason = None
            execution_log.append(f"PASSED: Final score {final_lead_score} >= {CONSENSUS_THRESHOLD}")

        logger.info(
            f"[{business_name}] Consensus result: passed={lead_should_proceed}, score={final_lead_score}, conflict={registry_website_conflict}"
        )

        # Create consensus result object
        consensus_result = ConsensusResult(
            lead_should_proceed=lead_should_proceed,
            name_match=name_match,
            address_match=address_match,
            address_mismatch_details=(
                None
                if address_match >= MIN_ADDRESS_MATCH
                else (
                    address_zip_mismatch_detail
                    or f"Address match {address_match:.2f} below {MIN_ADDRESS_MATCH}"
                )
            ),
            registry_website_conflict=registry_website_conflict,
            conflict_description=conflict_description,
            signal_scoring=signal_scoring,
            base_signal_score=base_signal_score,
            match_bonus=match_bonus,
            final_lead_score=final_lead_score,
            drop_reason=drop_reason,
            validation_notes=(
                f"Matching: name={name_match:.2f}, address={address_match:.2f} "
                f"(ZIP-first when both sides have ZIPs; else max of full-address "
                f"and street-core fuzzy)"
            ),
            recommended_follow_up="Proceed to enrichment" if lead_should_proceed else "Manual review recommended",
        )

    except Exception as e:
        logger.error(f"[{business_name}] Node 3 error: {e}", exc_info=True)
        execution_log.append(f"Node 3 error: {e}")
        lead_should_proceed = False
        final_lead_score = 0

    elapsed = time.time() - start_time
    logger.info(f"[{business_name}] Node 3 completed in {elapsed:.2f}s")

    return {
        "consensus_result": consensus_result,
        "name_match_confidence": consensus_result.name_match if consensus_result else 0.0,
        "address_match_confidence": consensus_result.address_match if consensus_result else 0.0,
        "lead_score": final_lead_score,
        "consensus_passed": lead_should_proceed,
        "dropped_reason": consensus_result.drop_reason if consensus_result else "Consensus failed",
        "execution_log": execution_log,
        "node_timestamps": {**state.get("node_timestamps", {}), "consensus": elapsed},
        "should_continue": lead_should_proceed,
    }
