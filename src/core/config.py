"""Configuration and environment setup."""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv('.env.local')


class Config:
    """Application configuration from environment variables."""

    # LLM Provider (ollama, google, or grok)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "grok")

    # API Keys (for Gemini/Google/Grok)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GROK_API_KEY: str = os.getenv("GROK_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    HUNTER_API_KEY: str = os.getenv("HUNTER_API_KEY", "")

    # Local Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8B")

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
        # If using Ollama locally, no API keys needed
        if cls.LLM_PROVIDER == "ollama":
            return

        # If using Grok, require API key or mocks
        if cls.LLM_PROVIDER == "grok" and not cls.GROK_API_KEY and not cls.USE_MOCKS:
            raise ValueError("GROK_API_KEY is required when LLM_PROVIDER is grok or USE_MOCKS must be true")

        # If using Google Gemini, require API key or mocks
        if cls.LLM_PROVIDER == "google" and not cls.GOOGLE_API_KEY and not cls.USE_MOCKS:
            raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER is google or USE_MOCKS must be true")

        if not cls.TAVILY_API_KEY and not cls.USE_MOCKS:
            raise ValueError("TAVILY_API_KEY is required or USE_MOCKS must be true")
        if not cls.HUNTER_API_KEY and not cls.USE_MOCKS:
            raise ValueError("HUNTER_API_KEY is required or USE_MOCKS must be true")

