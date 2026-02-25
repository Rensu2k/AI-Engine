"""Pydantic request/response schemas for the chat API."""

from pydantic import BaseModel, Field
from typing import Optional, Dict


class ChatRequest(BaseModel):
    """Request body for POST /ai/chat."""
    message: str = Field(..., min_length=1, max_length=1000, description="The user's message")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn conversation. Omit to start a new session.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What is the status of my document?",
                    "session_id": None,
                },
                {
                    "message": "PDID 001",
                    "session_id": "abc123-def456-ghi789",
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response body for POST /ai/chat."""
    reply: str = Field(..., description="The AI assistant's response")
    session_id: str = Field(..., description="Session ID (use this in the next request for multi-turn)")
    intent: str = Field(..., description="Classified intent of the user's message")
    confidence: float = Field(..., description="Confidence score of the classification")
    entities: Dict[str, str] = Field(default_factory=dict, description="Extracted entities (e.g., PDID)")


class TrainRequest(BaseModel):
    """Request body for POST /ai/train."""
    source: str = Field(
        default="csv",
        description="Training source: 'csv' (from ml_data/intent_training.csv) or 'database' (from training_data table)"
    )


class TrainResponse(BaseModel):
    """Response body for POST /ai/train."""
    status: str
    num_samples: int
    num_intents: int
    intents: list
    training_accuracy: float


class HealthResponse(BaseModel):
    """Response body for GET /ai/health."""
    status: str
    model_loaded: bool
    model_intents: list
