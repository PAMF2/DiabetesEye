"""Configuration for DiabetesEye CrewAI agent."""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Gemini API settings
    gemini_model: str = "gemini-2.0-flash"
    google_api_key: str = ""

    # LLM settings
    llm_model: str = "gemini-2.0-flash"
    llm_preferred: str = ""
    llm_allow_openai: bool = False
    openai_api_key: str = ""

    class Config:
        env_file = ".env"


config = Settings()