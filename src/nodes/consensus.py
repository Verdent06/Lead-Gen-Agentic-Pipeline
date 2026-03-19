"""Node 3: Triangulated Consensus & Deterministic Scoring (PURE PYTHON, NO LLM)."""

import logging
import time
from difflib import SequenceMatcher
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
    2. Address fuzzy match (must be >= MIN_ADDRESS_MATCH)
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
    logger.info("=== Node 3: Triangulated Consensus & Deterministic Scoring ===")

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

        # === STEP 1: Name Matching ===
        registry_name = registry_data.business_name or ""
        website_name = extracted_signals.business_name_from_site or registry_data.business_name or ""

        name_match = fuzzy_match(registry_name, website_name)
        logger.info(f"Name match: {registry_name} vs {website_name} = {name_match:.2f}")
        execution_log.append(f"Name match score: {name_match:.2f}")

        # === STEP 2: Address Matching ===
        registry_address = (
            f"{registry_data.business_address} {registry_data.city} {registry_data.state}".lower()
        )
        website_address = (
            extracted_signals.contact_information.get("address", "").lower()
            if extracted_signals.contact_information
            else ""
        )

        address_match = fuzzy_match(registry_address, website_address) if website_address else 0.5
        logger.info(f"Address match: {address_match:.2f}")
        execution_log.append(f"Address match score: {address_match:.2f}")

        # === STEP 3: Conflict Detection ===
        registry_website_conflict = False
        conflict_description = None

        # Check for critical mismatches
        if name_match < MIN_NAME_MATCH:
            registry_website_conflict = True
            conflict_description = f"Business name mismatch (similarity: {name_match:.2f} < {MIN_NAME_MATCH})"
            logger.warning(conflict_description)

        if address_match < MIN_ADDRESS_MATCH and website_address:
            registry_website_conflict = True
            conflict_description = f"Address mismatch (similarity: {address_match:.2f} < {MIN_ADDRESS_MATCH})"
            logger.warning(conflict_description)

        # === STEP 4: Calculate Signal Score ===
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

        # === STEP 5: Match Quality Bonus ===
        match_bonus = 0
        if name_match >= 0.95 and address_match >= 0.90:
            match_bonus = 20
            execution_log.append("+ 20 pts: Excellent name/address match bonus")
        elif name_match >= 0.85 and address_match >= 0.75:
            match_bonus = 10
            execution_log.append("+ 10 pts: Good name/address match bonus")

        # === STEP 6: Final Score Calculation ===
        final_lead_score = min(base_signal_score + match_bonus, 100)

        # === STEP 7: Pass/Fail Determination ===
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
            f"Consensus result: passed={lead_should_proceed}, score={final_lead_score}, conflict={registry_website_conflict}"
        )

        # Create consensus result object
        consensus_result = ConsensusResult(
            lead_should_proceed=lead_should_proceed,
            name_match=name_match,
            address_match=address_match,
            address_mismatch_details=None if address_match >= MIN_ADDRESS_MATCH else f"Address match {address_match:.2f} below {MIN_ADDRESS_MATCH}",
            registry_website_conflict=registry_website_conflict,
            conflict_description=conflict_description,
            signal_scoring=signal_scoring,
            base_signal_score=base_signal_score,
            match_bonus=match_bonus,
            final_lead_score=final_lead_score,
            drop_reason=drop_reason,
            validation_notes=f"Fuzzy matching: name={name_match:.2f}, address={address_match:.2f}",
            recommended_follow_up="Proceed to enrichment" if lead_should_proceed else "Manual review recommended",
        )

    except Exception as e:
        logger.error(f"Node 3 error: {e}", exc_info=True)
        execution_log.append(f"Node 3 error: {e}")
        lead_should_proceed = False
        final_lead_score = 0

    elapsed = time.time() - start_time
    logger.info(f"Node 3 completed in {elapsed:.2f}s")

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
