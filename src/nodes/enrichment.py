"""Node 4: Enrichment with Hunter.io Contact Information."""

import logging
import time
from src.models.state import LeadState
from src.models.schemas import HunterContact
from src.services.hunter_service import get_hunter_service
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


async def enrichment_node(state: LeadState) -> dict:
    """
    Node 4: Enrichment with Hunter.io.

    Takes verified lead from consensus and enriches with contact information.
    Searches Hunter.io for owner/decision maker emails.
    Identifies primary contact (likely owner or CEO).

    Args:
        state: Current LeadState from graph

    Returns:
        Updated state dict with enriched contacts
    """
    start_time = time.time()
    logger.info("=== Node 4: Enrichment ===")

    execution_log = state.get("execution_log", [])
    enriched_contacts = []
    primary_contact = None
    enrichment_success = False
    enrichment_error = None

    try:
        # Extract required data
        registry_data = state.get("registry_data")
        if not registry_data:
            raise ValueError("No registry data available for enrichment")

        website_url = state.get("website_url") or registry_data.official_website_url
        if not website_url:
            raise ValueError("No website URL available for Hunter.io domain enrichment")

        # Derive domain from business name or use known domain
        business_name = registry_data.business_name or ""
        owner_name = registry_data.owner_name or ""

        # Simple domain derivation (in production, would be more sophisticated)
        domain = urlparse(website_url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        execution_log.append(f"Searching Hunter.io for domain: {domain}")
        logger.info(f"Hunter.io search for: {domain}")

        # Call Hunter service
        hunter_service = await get_hunter_service()
        raw_contacts = await hunter_service.find_contacts(
            domain=domain,
            company_name=business_name,
            owner_name=owner_name,
        )

        if raw_contacts:
            execution_log.append(f"Found {len(raw_contacts)} contacts")
            logger.info(f"Hunter.io found {len(raw_contacts)} contacts")

            # Convert to HunterContact objects
            for contact in raw_contacts:
                hunter_contact = HunterContact(
                    first_name=contact.get("first_name"),
                    last_name=contact.get("last_name"),
                    email=contact.get("email"),
                    email_confidence=contact.get("email_confidence", 0.0),
                    phone=contact.get("phone"),
                    job_title=contact.get("job_title"),
                    department=contact.get("department"),
                    linkedin_profile=contact.get("linkedin_profile"),
                    is_owner_or_decision_maker=_is_decision_maker(
                        contact.get("job_title", ""), owner_name
                    ),
                    source="hunter.io",
                )
                enriched_contacts.append(hunter_contact)

            # Identify primary contact (owner/CEO if available)
            for contact in enriched_contacts:
                if contact.is_owner_or_decision_maker:
                    primary_contact = contact
                    break

            if not primary_contact and enriched_contacts:
                primary_contact = enriched_contacts[0]

            enrichment_success = True

        else:
            execution_log.append("No contacts found in Hunter.io")
            enrichment_error = "No contacts found"

    except Exception as e:
        logger.error(f"Node 4 error: {e}", exc_info=True)
        execution_log.append(f"Node 4 error: {e}")
        enrichment_error = str(e)

    elapsed = time.time() - start_time
    logger.info(f"Node 4 completed in {elapsed:.2f}s")

    return {
        "enrichment_data": enriched_contacts,
        "primary_contact": primary_contact,
        "enrichment_success": enrichment_success,
        "enrichment_error": enrichment_error,
        "execution_log": execution_log,
        "node_timestamps": {**state.get("node_timestamps", {}), "enrichment": elapsed},
    }


def _is_decision_maker(job_title: str, owner_name: str) -> bool:
    """Heuristic to determine if contact is likely owner/decision maker."""
    if not job_title:
        return False

    decision_keywords = [
        "owner",
        "ceo",
        "founder",
        "president",
        "principal",
        "managing partner",
        "executive",
    ]

    job_lower = job_title.lower()
    return any(keyword in job_lower for keyword in decision_keywords)
