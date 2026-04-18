"""Node 3: Triangulated Consensus & Deterministic Scoring (PURE PYTHON, NO LLM)."""

import logging
import re
import time
from difflib import SequenceMatcher
from typing import Dict, Optional, Set

from src.models.state import LeadState
from src.models.schemas import ConsensusResult

logger = logging.getLogger(__name__)

# Dynamic signal scoring (Node 3): thesis-defined signals from Node 2 carry confidence 0–1.
POINTS_PER_DETECTED_SIGNAL = 25  # Weight for a fully confident detected signal
MAX_BASE_SIGNAL_SCORE = 90  # Cap on thesis-signal contribution before registry/name bonuses

MIN_NAME_MATCH = 0.60  # Minimum fuzzy match for business names
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


def _registry_state_abbr(state: Optional[str]) -> Optional[str]:
    """Return lowercase USPS-style state abbreviation, or None if not parseable."""
    if state is None or not str(state).strip():
        return None
    s = re.sub(r"\s+", " ", str(state).strip().lower())
    if len(s) == 2 and s in _US_STATE_ABBRS:
        return s
    for full, abbr in _US_STATE_NAME_TO_ABBR.items():
        if re.search(rf"\b{re.escape(full)}\b", s):
            return abbr
    return None


def _state_abbrs_in_text(text: str) -> Set[str]:
    """
    US states mentioned in free text (e.g. pipeline ``location``).

    Uses full state names plus ``City, ST``-style abbreviations only. We do not
    scan for bare 2-letter tokens (avoids false positives like "in" / "or").
    """
    if not text or not str(text).strip():
        return set()
    s = str(text).lower()
    found: Set[str] = set()
    for full, abbr in _US_STATE_NAME_TO_ABBR.items():
        if re.search(rf"\b{re.escape(full)}\b", s):
            found.add(abbr)
    for m in re.finditer(r"(?:^|[,])\s*([a-z]{2})\b", s):
        ab = m.group(1)
        if ab in _US_STATE_ABBRS:
            found.add(ab)
    return found


def _pipeline_location_state_conflict(
    registry_state: Optional[str],
    pipeline_location: Optional[str],
) -> tuple[bool, Optional[str]]:
    """
    If the registry lists a US state and the pipeline location also implies
    state(s), flag a conflict when they disagree. Intentionally ignores
    branch vs HQ street/ZIP differences on the website.
    """
    reg = _registry_state_abbr(registry_state)
    loc = _state_abbrs_in_text(pipeline_location or "")
    if reg and loc and reg not in loc:
        loc_u = ", ".join(sorted({x.upper() for x in loc}))
        return True, f"Registry state {reg.upper()} vs pipeline location state(s) [{loc_u}]"
    return False, None


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


async def consensus_node(state: LeadState) -> dict:
    """
    Node 3: Triangulated Consensus & Deterministic Scoring.

    **PURE DETERMINISTIC PYTHON - NO LLM CALLS.**

    Compares registry data (Node 1) against website signals (Node 2).
    Validates:
    1. Business name fuzzy match (must be >= MIN_NAME_MATCH)
    2. Optional pipeline location vs registry state consistency (no street/ZIP
       fuzzy matching — avoids HQ vs branch false negatives)
    3. No conflicting data between registry and website

    Calculates final_lead_score (0-100) based on:
    - Thesis signal rows from Node 2 (25 × confidence per detected signal, base capped at 90)
    - Name match bonus
    - Registry status validation bonus

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

        # === STEP 2: Location sanity (registry state vs pipeline location only) ===
        # Street/ZIP fuzzy matching vs website was removed: HQ vs branch addresses
        # produced false rejections. ``address_match`` is kept for the schema as 1.0
        # when we are not using website address comparison.
        pipeline_location = state.get("location") or ""
        state_conflict, state_conflict_detail = _pipeline_location_state_conflict(
            getattr(registry_data, "state", None),
            pipeline_location,
        )
        address_match = 1.0
        reg_st = _registry_state_abbr(getattr(registry_data, "state", None))
        loc_st = sorted(_state_abbrs_in_text(pipeline_location))
        logger.info(
            f"[{business_name}] Location check: registry_state={reg_st or '(none)'} | "
            f"pipeline_location={_truncate_for_log(pipeline_location, 200)} | "
            f"states_in_location={loc_st or '[]'} | conflict={state_conflict}"
        )
        execution_log.append(
            "Address fuzzy matching disabled (name + optional state-vs-location only)"
        )
        if state_conflict and state_conflict_detail:
            execution_log.append(f"State/location flag: {state_conflict_detail}")

        # === STEP 3: HARD FAIL - Industry Verification ===
        # If the website is NOT in the target industry (HVAC distribution), immediately fail
        if not extracted_signals.is_target_industry:
            logger.warning(f"[{business_name}] HARD FAIL: Irrelevant industry. Evidence: {extracted_signals.industry_evidence}")
            execution_log.append(f"REJECTED: Irrelevant Industry - {extracted_signals.industry_evidence}")
            
            consensus_result = ConsensusResult(
                lead_should_proceed=False,
                name_match=name_match,
                address_match=1.0,
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

        if state_conflict and state_conflict_detail:
            registry_website_conflict = True
            msg = f"State/location mismatch: {state_conflict_detail}"
            if conflict_description:
                conflict_description = f"{conflict_description}; {msg}"
            else:
                conflict_description = msg
            logger.warning(f"[{business_name}] {msg}")

        # === STEP 5: Calculate Signal Score (thesis signals × confidence, capped) ===
        signal_scoring: Dict[str, int] = {}
        sum_signal_points = 0
        used_keys: Set[str] = set()

        for i, sig in enumerate(extracted_signals.signals or []):
            if not sig.detected:
                continue
            pts = int(round(POINTS_PER_DETECTED_SIGNAL * float(sig.confidence)))
            base_name = (sig.signal_name or "").strip() or f"signal_{i}"
            key = base_name
            suffix = 1
            while key in used_keys:
                suffix += 1
                key = f"{base_name}_{suffix}"
            used_keys.add(key)
            signal_scoring[key] = pts
            sum_signal_points += pts
            ev = (sig.evidence or "")[:80]
            execution_log.append(
                f"+ {pts} pts: '{base_name}' (conf={sig.confidence:.2f}) — {ev}{'...' if len(sig.evidence or '') > 80 else ''}"
            )

        base_signal_score = min(sum_signal_points, MAX_BASE_SIGNAL_SCORE)
        if sum_signal_points > MAX_BASE_SIGNAL_SCORE:
            execution_log.append(
                f"Base signal score capped: sum_signals={sum_signal_points} → {base_signal_score} (max {MAX_BASE_SIGNAL_SCORE})"
            )

        # === STEP 6: Match Quality Bonus (name only; address fuzzy removed) ===
        match_bonus = 0
        if name_match >= 0.95:
            match_bonus = 20
            execution_log.append("+ 20 pts: Excellent name match bonus")
        elif name_match >= 0.85:
            match_bonus = 10
            execution_log.append("+ 10 pts: Good name match bonus")

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
                state_conflict_detail if state_conflict else None
            ),
            registry_website_conflict=registry_website_conflict,
            conflict_description=conflict_description,
            signal_scoring=signal_scoring,
            base_signal_score=base_signal_score,
            match_bonus=match_bonus,
            final_lead_score=final_lead_score,
            drop_reason=drop_reason,
            validation_notes=(
                f"Matching: name={name_match:.2f}; address_line_match disabled (1.0); "
                f"registry vs pipeline state check: {'fail' if state_conflict else 'ok'}"
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
