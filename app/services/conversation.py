"""
Conversation engine — orchestrates the full chat pipeline.

Handles:
1. Session management (create/resume multi-turn sessions)
2. Intent classification
3. Entity extraction
4. DTS API lookup
5. Response generation (LLM or template fallback)
6. Chat logging
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session as DBSession

from app.db.models import Session, ChatLog
from app.ml.intent_classifier import IntentClassifier
from app.ml.entity_extractor import extract_entities
from app.services.dts_client import get_document
from app.services.response_generator import generate_response
from app.services.chat_logger import log_message
from app.services.llm_client import generate_llm_response
from app.services import rag_service
from app.config import settings


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
    language: str = "en",
) -> Dict[str, Any]:
    """
    Process a user message through the full AI pipeline.

    Pipeline:
    1. Get/create session
    2. Classify intent
    3. Extract entities (PDID)
    4. Check session context for pending PDID from previous turn
    5. Fetch document from DTS if PDID available
    6. Generate response (LLM if enabled, template as fallback)
    7. Log messages

    Args:
        db: Database session
        message: Raw user message
        session_id: Optional session ID for multi-turn context
        language: Language hint (kept for API compatibility)

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

    # 5b. RAG retrieval — fetch relevant ELA document context
    rag_context = None
    if settings.USE_RAG and rag_service.is_ready():
        # Only retrieve RAG context for general queries, unknown intents, or 
        # document queries that didn't have a specific PDID match.
        # Skip for explicit tracking commands where we already have the document, greetings, etc.
        if not document and intent in ("lgu_query", "tourism_query", "unknown", "document_status", "follow_up"):
            rag_context = rag_service.retrieve_context(
                query=message,
                top_k=settings.RAG_TOP_K,
            )

    # 6. Generate response — try LLM first, fall back to templates
    reply = None
    if settings.USE_LLM:
        try:
            reply = await generate_llm_response(intent, entities, document, context, rag_context=rag_context, user_message=message)
        except Exception as e:
            print(f"LLM generation failed, falling back to template: {e}")

    if not reply:
        # Fallback to template generator
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

import json

async def stream_message(
    db: DBSession,
    message: str,
    session_id: Optional[str] = None,
    language: str = "en",
):
    """
    Process a user message and yield Server-Sent Events (SSE).
    """
    # 1. Session setup
    session = get_or_create_session(db, session_id)
    intent, confidence = classifier.predict(message)
    entities = extract_entities(message)

    context = dict(session.context) if session.context else {}
    pending_intent = context.get("pending_intent")

    if "pdid" in entities and pending_intent == "document_status":
        intent = "follow_up"
        update_session_context(db, session, "pending_intent", None)
    if intent == "document_status" and "pdid" not in entities:
        update_session_context(db, session, "pending_intent", "document_status")
    if "pdid" not in entities and context.get("last_pdid"):
        if intent in ("document_status", "follow_up"):
            entities["pdid"] = context["last_pdid"]
    if "pdid" in entities:
        update_session_context(db, session, "last_pdid", entities["pdid"])

    document = None
    if "pdid" in entities:
        document = await get_document(entities["pdid"])

    rag_context = None
    if settings.USE_RAG and rag_service.is_ready() and not document:
        if intent in ("lgu_query", "tourism_query", "unknown", "document_status", "follow_up"):
            rag_context = rag_service.retrieve_context(query=message, top_k=settings.RAG_TOP_K)

    # First yield the metadata (intent, entities, sessionid)
    metadata = {
        "session_id": session.id,
        "intent": intent,
        "confidence": round(confidence, 4),
        "entities": entities,
    }
    yield f"data: {json.dumps(metadata)}\n\n"

    full_reply = ""
    
    # Check if we can stream from LLM
    if settings.USE_LLM:
        from app.services.llm_client import generate_llm_response_stream
        
        # This returns an httpx.Response object set to stream
        response_stream = await generate_llm_response_stream(
            intent, entities, document, context, rag_context=rag_context, user_message=message
        )
        
        if response_stream:
            try:
                # Iterate asynchronously over the response stream line-by-line
                async for chunk in response_stream.aiter_lines():
                    if chunk:
                        try:
                            # Ollama returns JSON lines, we need to extract the "response" key 
                            # or just pass the text forward. 
                            # Since the frontend only accepts {"text": "..."} we need to parse it if it's JSON from Ollama
                            chunk_data = json.loads(chunk)
                            text_chunk = chunk_data.get("response", chunk)
                        except json.JSONDecodeError:
                            text_chunk = chunk
                            
                        full_reply += text_chunk
                        # Send text chunks as they arrive
                        yield f"data: {json.dumps({'text': text_chunk})}\n\n"
            except Exception as e:
                print(f"Error streaming LLM response: {e}")
            finally:
                # Always safely close the httpx stream
                await response_stream.aclose()
                
    # If no LLM streaming occurred/succeeded, fallback to template
    if not full_reply:
        full_reply = generate_response(intent, entities, document, context)
        # Yield the full template response at once
        yield f"data: {json.dumps({'text': full_reply})}\n\n"

    # Send completion event
    yield f"data: [DONE]\n\n"
    
    # Log the complete interaction
    log_message(db, session.id, "user", message, intent, confidence, entities)
    log_message(db, session.id, "bot", full_reply)
