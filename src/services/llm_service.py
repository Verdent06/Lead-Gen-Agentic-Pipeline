"""Google Gemini LLM service with structured output support."""

import json
from typing import Type, TypeVar, Optional
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from src.config import Config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """Wrapper for Google Gemini API with Pydantic schema enforcement."""

    def __init__(self):
        """Initialize Gemini LLM client."""
        Config.validate()
        self.client = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            google_api_key=Config.GOOGLE_API_KEY,
        )

    async def extract_structured(
        self,
        prompt: str,
        response_model: Type[T],
        context: str = "",
    ) -> Optional[T]:
        """
        Extract structured data from prompt using Gemini with Pydantic validation.

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
            # Construct system prompt for structured extraction
            system_prompt = f"""You are an expert data extraction agent.
Extract information from the provided content and return it in the specified JSON format.
Ensure all fields are validated and match the schema.

{context}"""

            full_prompt = f"""{system_prompt}

Content to extract from:
{prompt}"""

            # Use Gemini with structured output
            response = await self.client.ainvoke(full_prompt)

            # Parse JSON from response
            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content

            extracted_json = json.loads(json_str)
            result = response_model.model_validate(extracted_json)
            logger.info(f"Successfully extracted {response_model.__name__}")
            return result

        except Exception as e:
            logger.error(f"Failed to extract {response_model.__name__}: {e}")
            return None

    def _mock_response(self, response_model: Type[T]) -> T:
        """Generate mock response matching Pydantic schema (for testing)."""
        # This would be populated with real mock data in actual implementation
        # For now, return a basic instance
        try:
            # Try to create with no args (schema-dependent)
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
