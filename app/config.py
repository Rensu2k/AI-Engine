"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Settings loaded from .env file or environment variables."""

    # Database
    DATABASE_URL: str = "mysql+pymysql://root:password@localhost:3306/dts_ai_engine"

    # DTS Backend API
    DTS_API_BASE_URL: str = "http://localhost:8080"
    DTS_MOCK_MODE: bool = True

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ML
    MODEL_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_models")
    TRAINING_DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_data")
    CONFIDENCE_THRESHOLD: float = 0.4

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
