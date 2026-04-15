"""Hunter.io API service for contact enrichment."""

import logging
from typing import List, Dict, Any, Optional
import httpx
import json
from src.config import Config

logger = logging.getLogger(__name__)


class HunterService:
    """Async wrapper for Hunter.io domain search and email finder API."""

    BASE_URL = "https://api.hunter.io/v2"

    def __init__(self, api_key: str = ""):
        """Initialize Hunter service."""
        self.api_key = api_key or Config.HUNTER_API_KEY

    async def find_contacts(
        self,
        domain: str,
        company_name: Optional[str] = None,
        owner_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find email addresses and contacts for a domain/company.

        Args:
            domain: Company domain (e.g., example.com)
            company_name: Company name for additional filtering
            owner_name: Owner/CEO name to match

        Returns:
            List of contacts with email, job title, and confidence scores
        """
        if Config.USE_MOCKS:
            logger.debug(f"Using mock Hunter results for domain: {domain}")
            return self._mock_contacts(domain, owner_name)

        try:
            async with httpx.AsyncClient() as client:
                # First, search for domain info
                params = {
                    "domain": domain,
                    "limit": 10,
                }

                response = await client.get(
                    f"{self.BASE_URL}/domain-search",
                    params=params,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30.0,
                )
                response.raise_for_status()

                result = response.json()
                contacts = result.get("data", {}).get("employees", [])

                # Filter and enrich contacts
                enriched_contacts = []
                for contact in contacts:
                    enriched_contacts.append(
                        {
                            "first_name": contact.get("first_name"),
                            "last_name": contact.get("last_name"),
                            "email": contact.get("email"),
                            "email_confidence": contact.get("confidence"),
                            "job_title": contact.get("title"),
                            "department": contact.get("department"),
                            "linkedin_profile": contact.get("linkedin_url"),
                        }
                    )

                logger.info(
                    f"Hunter.io found {len(enriched_contacts)} contacts for {domain}"
                )
                return enriched_contacts

        except Exception as e:
            logger.error(f"Hunter.io search failed for domain '{domain}': {e}")
            return []

    def _mock_contacts(
        self, domain: str, owner_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return hardcoded mock contacts for testing."""
        return [
            {
                "first_name": "Jordan",
                "last_name": "Lee",
                "email": "jordan.lee@example-company-corp.test",
                "email_confidence": 0.98,
                "job_title": "Owner/CEO",
                "department": "Management",
                "linkedin_profile": "https://www.linkedin.com/in/example-company-corp",
            },
            {
                "first_name": "Morgan",
                "last_name": "Taylor",
                "email": "morgan.taylor@example-company-corp.test",
                "email_confidence": 0.95,
                "job_title": "Operations Manager",
                "department": "Operations",
                "linkedin_profile": "https://www.linkedin.com/in/example-ops",
            },
            {
                "first_name": "Casey",
                "last_name": "Nguyen",
                "email": "casey.nguyen@example-company-corp.test",
                "email_confidence": 0.92,
                "job_title": "Sales Director",
                "department": "Sales",
                "linkedin_profile": None,
            },
        ]


# Global Hunter service instance
_hunter_service: Optional[HunterService] = None


async def get_hunter_service() -> HunterService:
    """Get or initialize the global Hunter service."""
    global _hunter_service
    if _hunter_service is None:
        _hunter_service = HunterService()
    return _hunter_service
