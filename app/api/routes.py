"""API routes for the DTS AI Engine."""

import os
import re
import io
import edge_tts
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session as DBSession

from app.db.database import get_db
from app.db.models import TrainingData
from app.schemas.chat import (
    ChatRequest, ChatResponse,
    TrainRequest, TrainResponse,
    HealthResponse,
    TTSRequest,
)
from app.services.conversation import process_message, classifier
from app.config import settings

router = APIRouter(prefix="/api", tags=["AI Engine"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: DBSession = Depends(get_db)):
    """
    Main chat endpoint.

    Send a message and get an AI-powered response about document status.
    Pass `session_id` from a previous response to continue a multi-turn conversation.
    """
    try:
        result = await process_message(
            db=db,
            message=request.message,
            session_id=request.session_id,
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest = TrainRequest(), db: DBSession = Depends(get_db)):
    """
    Retrain the intent classifier.

    Source options:
    - `csv`: Train from ml_data/intent_training.csv
    - `database`: Train from the training_data table in the database
    """
    try:
        data = None
        csv_path = None

        if request.source == "database":
            # Load training data from database
            records = db.query(TrainingData).all()
            if not records:
                raise HTTPException(
                    status_code=400,
                    detail="No training data found in database. Add data to the training_data table first."
                )
            data = [(r.text, r.intent) for r in records]

        elif request.source == "csv":
            csv_path = os.path.join(settings.TRAINING_DATA_DIR, "intent_training.csv")
            if not os.path.exists(csv_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Training CSV not found at {csv_path}"
                )

        else:
            raise HTTPException(status_code=400, detail="Invalid source. Use 'csv' or 'database'.")

        stats = classifier.train(csv_path=csv_path, data=data)

        return TrainResponse(
            status="success",
            num_samples=stats["num_samples"],
            num_intents=stats["num_intents"],
            intents=stats["intents"],
            training_accuracy=stats["training_accuracy"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health():
    """Check engine health and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=classifier.is_loaded,
        model_intents=classifier.classes,
    )


# Allowed TTS voices
ALLOWED_VOICES = {
    "en-US-GuyNeural",         # English male
    "fil-PH-AngeloNeural",     # Filipino male
}


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting and emojis from text so TTS reads cleanly."""
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove all emojis and other pictographic symbols
    text = re.sub(
        r'[\U00010000-\U0010FFFF'   # Supplementary multilingual plane (most emojis)
        r'\U00002600-\U000027BF'    # Misc symbols (☀, ✅, etc.)
        r'\U0001F300-\U0001F9FF'    # Emoji block
        r'\u2190-\u21FF]',          # Arrows (←, →)
        '', text, flags=re.UNICODE
    )
    # Expand common abbreviations for natural TTS reading
    text = re.sub(r'\bNo\.(?=\s|$)', 'Number', text)
    # Remove bullet points
    text = text.replace("• ", "")
    # Convert newlines to natural pauses
    text = re.sub(r'\n+', '. ', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech audio (MP3).

    Accepts the AI reply text and returns an MP3 audio stream.
    Supports English and Filipino voices.

    Available voices:
    - `en-US-JennyNeural` (English, female - default)
    - `en-US-AriaNeural` (English, female)
    - `en-US-GuyNeural` (English, male)
    - `fil-PH-BlessicaNeural` (Filipino, female)
    - `fil-PH-AngeloNeural` (Filipino, male)
    """
    # Validate voice
    voice = request.voice
    if voice not in ALLOWED_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice '{voice}'. Allowed: {', '.join(sorted(ALLOWED_VOICES))}"
        )

    # Strip markdown from text
    clean_text = _strip_markdown(request.text)

    if not clean_text:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning.")

    try:
        # Generate audio using edge-tts
        communicate = edge_tts.Communicate(clean_text, voice)
        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=tts_output.mp3"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

