"""Configuration and environment setup."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration from environment variables."""

    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    HUNTER_API_KEY: str = os.getenv("HUNTER_API_KEY", "")

    # LLM Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # Consensus Thresholds
    CONSENSUS_SCORE_THRESHOLD: int = int(os.getenv("CONSENSUS_SCORE_THRESHOLD", "70"))
    MATCH_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("MATCH_CONFIDENCE_THRESHOLD", "0.85")
    )

    # Feature Flags
    USE_MOCKS: bool = os.getenv("USE_MOCKS", "false").lower() == "true"

    @classmethod
    def validate(cls) -> None:
        """Validate that all required config is present."""
        if not cls.GOOGLE_API_KEY and not cls.USE_MOCKS:
            raise ValueError("GOOGLE_API_KEY is required or USE_MOCKS must be true")
        if not cls.TAVILY_API_KEY and not cls.USE_MOCKS:
            raise ValueError("TAVILY_API_KEY is required or USE_MOCKS must be true")
        if not cls.HUNTER_API_KEY and not cls.USE_MOCKS:
            raise ValueError("HUNTER_API_KEY is required or USE_MOCKS must be true")
