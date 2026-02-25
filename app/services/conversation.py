"""
Conversation engine — orchestrates the full chat pipeline.

Handles:
1. Session management (create/resume multi-turn sessions)
2. Intent classification
3. Entity extraction
4. DTS API lookup
5. Response generation
6. Chat logging
"""

import uuid
import json
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from sqlalchemy.orm import Session as DBSession

from app.db.models import Session, ChatLog
from app.ml.intent_classifier import IntentClassifier
from app.ml.entity_extractor import extract_entities
from app.services.dts_client import get_document
from app.services.response_generator import generate_response
from app.services.chat_logger import log_message


# Global classifier instance (loaded at startup)
classifier = IntentClassifier()


def get_or_create_session(db: DBSession, session_id: Optional[str] = None) -> Session:
    """Get an existing session or create a new one."""
    if session_id:
        session = db.query(Session).filter(Session.id == session_id).first()
        if session:
            session.last_active = datetime.utcnow()
            db.commit()
            return session

    # Create new session
    new_session = Session(
        id=str(uuid.uuid4()),
        context={},
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def update_session_context(db: DBSession, session: Session, key: str, value: Any):
    """Update a key in the session context JSON."""
    ctx = dict(session.context) if session.context else {}
    ctx[key] = value
    session.context = ctx
    db.commit()


async def process_message(
    db: DBSession,
    message: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a user message through the full AI pipeline.

    Pipeline:
    1. Get/create session
    2. Classify intent
    3. Extract entities (PDID)
    4. Check session context for pending PDID from previous turn
    5. Fetch document from DTS if PDID available
    6. Generate response
    7. Log messages

    Args:
        db: Database session
        message: Raw user message
        session_id: Optional session ID for multi-turn context

    Returns:
        Dict with reply, session_id, intent, confidence, entities
    """
    # 1. Session
    session = get_or_create_session(db, session_id)

    # 2. Classify intent
    intent, confidence = classifier.predict(message)

    # 3. Extract entities
    entities = extract_entities(message)

    # 4. Multi-turn context: check if we're waiting for a PDID
    context = dict(session.context) if session.context else {}
    pending_intent = context.get("pending_intent")

    # If the user provides a PDID (or just a number) and we were waiting for one
    if "pdid" in entities and pending_intent == "document_status":
        intent = "follow_up"
        # Clear the pending state
        update_session_context(db, session, "pending_intent", None)

    # If intent is document_status but no PDID, remember we're asking for one
    if intent == "document_status" and "pdid" not in entities:
        update_session_context(db, session, "pending_intent", "document_status")

    # Also check if there's a PDID in context from a previous message
    if "pdid" not in entities and context.get("last_pdid"):
        # Only use context PDID if the current intent is related
        if intent in ("document_status", "follow_up"):
            entities["pdid"] = context["last_pdid"]

    # Save PDID to context for future reference
    if "pdid" in entities:
        update_session_context(db, session, "last_pdid", entities["pdid"])

    # 5. Fetch document from DTS if we have a PDID
    document = None
    if "pdid" in entities:
        document = await get_document(entities["pdid"])

    # 6. Generate response
    reply = generate_response(intent, entities, document, context)

    # 7. Log messages
    log_message(db, session.id, "user", message, intent, confidence, entities)
    log_message(db, session.id, "bot", reply)

    return {
        "reply": reply,
        "session_id": session.id,
        "intent": intent,
        "confidence": round(confidence, 4),
        "entities": entities,
    }
