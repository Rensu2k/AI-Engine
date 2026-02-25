"""
DTS AI Engine — FastAPI Application

Main entry point for the AI-powered Document Tracking Assistant.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router
from app.db.database import engine
from app.db.models import Base
from app.services.conversation import classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # --- Startup ---
    print("🚀 Starting DTS AI Engine...")

    # Create database tables if they don't exist
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables ready")

    # Load ML model
    csv_path = os.path.join(settings.TRAINING_DATA_DIR, "intent_training.csv")

    if classifier.load():
        print(f"✅ Intent classifier loaded ({len(classifier.classes)} intents)")
    elif os.path.exists(csv_path):
        # No saved model, but training data exists — auto-train
        print("🔄 No saved model found. Training from CSV...")
        stats = classifier.train(csv_path=csv_path)
        print(f"✅ Trained intent classifier: {stats['num_samples']} samples, "
              f"{stats['num_intents']} intents, {stats['training_accuracy']*100:.1f}% accuracy")
    else:
        print("⚠️  No model or training data found. Train the model via POST /ai/train")

    yield

    # --- Shutdown ---
    print("👋 Shutting down DTS AI Engine...")


app = FastAPI(
    title="DTS AI Engine",
    description=(
        "AI-powered Document Tracking Assistant. "
        "Provides a conversational interface for checking document status via the DTS system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — confirms the engine is running."""
    return {
        "name": "DTS AI Engine",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/ai/health",
    }
