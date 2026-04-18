"""LLM service with support for multiple providers (Grok, Ollama, Gemini)."""

import json
from typing import Type, TypeVar, Optional
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from src.config import Config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """Wrapper for LLM API with Pydantic schema enforcement (Grok, Ollama, or Gemini)."""

    def __init__(self):
        """Initialize LLM client based on configured provider."""
        Config.validate()
        
        if Config.LLM_PROVIDER == "grok":
            model_label = "grok-4.20-0309-non-reasoning"
            self.client = ChatOpenAI(
                model=model_label,
                api_key=Config.GROK_API_KEY,
                base_url="https://api.x.ai/v1",
                temperature=Config.LLM_TEMPERATURE,
            )
        elif Config.LLM_PROVIDER == "ollama":
            model_label = Config.OLLAMA_MODEL
            self.client = ChatOllama(
                model=Config.OLLAMA_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=Config.LLM_TEMPERATURE,
            )
        else:  # Default to Google Gemini
            model_label = Config.LLM_MODEL
            self.client = ChatGoogleGenerativeAI(
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE,
                google_api_key=Config.GOOGLE_API_KEY,
            )

        self.provider = Config.LLM_PROVIDER
        extra = ""
        if Config.LLM_PROVIDER == "ollama":
            extra = f" at {Config.OLLAMA_BASE_URL}"
        logger.info(f"Initializing {Config.LLM_PROVIDER} LLM{extra} (model: {model_label})")

    async def extract_structured(
        self,
        prompt: str,
        response_model: Type[T],
        context: str = "",
    ) -> Optional[T]:
        """
        Extract structured data from prompt with Pydantic validation.
        
        Uses .with_structured_output() for direct schema enforcement.

        Args:
            prompt: The user prompt
            response_model: Pydantic model class for response validation
            context: Additional context for the extraction

        Returns:
            Parsed Pydantic model instance or None if extraction fails
        """
        if Config.USE_MOCKS:
            logger.debug(f"Using mock LLM response for {response_model.__name__}")
            return self._mock_response(response_model)

        try:
            # Get the JSON schema for the response model
            schema_dict = response_model.model_json_schema()
            schema_string = json.dumps(schema_dict, indent=2)

            # Construct system prompt for structured extraction
            system_prompt = f"""You are an expert data extraction agent.
Extract information from the provided content and return it in VALID JSON format.
You MUST return a JSON object that strictly adheres to the following JSON schema:

{schema_string}

Ensure all fields are validated and match the schema exactly.
Do not return an array unless the schema explicitly specifies an array type.
Return ONLY valid JSON with no markdown code blocks, no explanations, no additional text.

{context}"""

            full_prompt = f"""{system_prompt}

Content to extract from:
{prompt}"""

            # Use LLM with structured output
            # .with_structured_output() enforces schema directly, bypassing string parsing
            llm_with_schema = self.client.with_structured_output(response_model)
            result = await llm_with_schema.ainvoke(full_prompt)
            
            logger.info(f"Successfully extracted {response_model.__name__}")
            return result

        except Exception as e:
            logger.error(f"Failed to extract {response_model.__name__}: {e}", exc_info=True)
            return None

    def _mock_response(self, response_model: Type[T]) -> T:
        """Generate mock response matching Pydantic schema (for testing)."""
        from src.models.schemas import WebsiteSignals, DetectedSignal

        if response_model is WebsiteSignals:
            return WebsiteSignals(
                website_url="https://example-hvac-mock.test",
                website_reachable=True,
                signals=[
                    DetectedSignal(
                        signal_name="modern_b2b_portal",
                        detected=True,
                        confidence=0.9,
                        evidence="Contractor login and online ordering referenced on homepage.",
                    ),
                    DetectedSignal(
                        signal_name="recent_acquisition",
                        detected=False,
                        confidence=0.25,
                        evidence="No press or site copy about M&A in provided markdown.",
                    ),
                    DetectedSignal(
                        signal_name="succession_or_multigenerational",
                        detected=True,
                        confidence=0.78,
                        evidence="Family-owned since 1972; second generation leadership mentioned.",
                    ),
                ],
                business_name_from_site="Example Company Corp",
                is_target_industry=True,
                industry_evidence="HVAC wholesale distributor serving contractors (mock).",
                extraction_confidence=0.86,
            )
        try:
            return response_model()
        except Exception:
            return None


# Global LLM service instance
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """Get or initialize the global LLM service."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
