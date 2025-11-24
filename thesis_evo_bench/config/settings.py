"""Settings and configuration management using environment variables."""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    llm_api_key: str
    llm_base_url: str = "https://api.deepseek.com/v1"
    default_model: str = "deepseek-reasoner"

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv(
                "LLM_BASE_URL",
                "https://api.deepseek.com/v1",
            ),
            default_model=os.getenv("DEFAULT_MODEL", "deepseek-reasoner"),
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()

