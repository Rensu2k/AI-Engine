"""Client for the central LLM Service."""

import httpx
import logging
from typing import Dict, Any, Optional

from app.config import settings
from app.services.response_generator import _format_document_status

logger = logging.getLogger(__name__)


async def generate_llm_response(
    intent: str,
    entities: Dict[str, str],
    document: Optional[Dict[str, Any]] = None,
    context: dict = None,
    rag_context: Optional[str] = None,
    user_message: str = "",
) -> Optional[str]:
    """
    Call the external LLM Service to generate a conversational response.
    Returns None if the LLM generation fails, allowing fallback to templates.
    """
    # Build prompt based on intent
    prompt = _build_prompt(intent, entities, document, context, rag_context, user_message)
    if not prompt:
        return None  # Will fall back to template rules (e.g., asking for PDID)

    system_prompt = (
        "You are the DTS AI Assistant built by Clarence Buenaflor, Jester Pastor & Mharjade Enario. "
        "You help users track their documents in the Document Tracking System. "
        "Be friendly, helpful, and concise. "
        "If document data is provided, use it to accurately answer the user's question. "
        "Do NOT hallucinate document statuses — only use the data provided."
    )

    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            response = await client.post(
                f"{settings.LLM_SERVICE_URL}/api/generate",
                json={
                    "prompt": prompt,
                    "system_prompt": system_prompt
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response")
    except Exception as e:
        logger.error(f"Error calling LLM Service: {e}")
        return None


async def generate_llm_response_stream(
    intent: str,
    entities: Dict[str, str],
    document: Optional[Dict[str, Any]] = None,
    context: dict = None,
    rag_context: Optional[str] = None,
    user_message: str = "",
) -> httpx.Response:
    """
    Call the external LLM Service to generate a conversational response and stream it.
    This expects the LLM Service's /api/generate endpoint to support streaming with `stream: True`.
    If the LLM Service doesn't support streaming, it will just yield the whole response at once.
    """
    # Build prompt based on intent
    prompt = _build_prompt(intent, entities, document, context, rag_context, user_message)
    if not prompt:
        return None  # Will fall back to template rules (e.g., asking for PDID)

    system_prompt = (
        "You are the DTS AI Assistant built by Clarence Buenaflor, Jester Pastor & Mharjade Enario. "
        "You help users track their documents in the Document Tracking System. "
        "Be friendly, helpful, and concise. "
        "If document data is provided, use it to accurately answer the user's question. "
        "Do NOT hallucinate document statuses — only use the data provided."
    )

    try:
        # We don't use 'async with httpx.AsyncClient()' here because we need to yield from the stream
        client = httpx.AsyncClient(timeout=35.0)
        request = client.build_request(
            "POST",
            f"{settings.LLM_SERVICE_URL}/api/generate",
            json={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "stream": True  # Request streaming if supported
            }
        )
        # Yield the response so the caller can stream the content
        return await client.send(request, stream=True)
    except Exception as e:
        logger.error(f"Error calling LLM Service Stream: {e}")
        return None


def _build_prompt(
    intent: str,
    entities: Dict[str, str],
    document: Optional[Dict[str, Any]],
    context: dict,
    rag_context: Optional[str] = None,
    user_message: str = "",
) -> Optional[str]:
    """Build the prompt string sent to the LLM."""
    if intent in ("document_status", "follow_up"):
        if not document and "pdid" in entities:
            return (
                f"The user asked about document Tracking No. {entities['pdid']}, "
                f"but no such document was found in the database. "
                f"Politely inform them it could not be found."
            )

        if document:
            doc_info = _format_document_status(document)
            return (
                f"The user is asking for the status of their document.\n\n"
                f"Here is the raw tracking data from the database:\n{doc_info}\n\n"
                f"Please provide a natural, conversational response telling the user "
                f"the status of their document. You can include the bulleted information "
                f"if it's helpful, but frame it naturally."
            )

        # If intent is document_status but no PDID/document, the template handles
        # asking for PDID best, so return None to fall back to template.
        return None

    if intent == "help":
        return (
            "The user is asking for help or what you can do. Briefly explain that you "
            "can help them track documents if they provide a Tracking Number (PDID)."
        )

    if intent == "complaint":
        return (
            "The user is expressing a complaint or frustration. Be empathetic, apologize "
            "for any inconvenience, and suggest they visit the DTS office or provide their "
            "Tracking No. so you can check on it."
        )

    # RAG-powered general query — answer using retrieved ELA document context
    if rag_context:
        question = user_message or "a question"
        return (
            f"The user asked: \"{question}\"\n\n"
            f"Use ONLY the following excerpts from our official ELA 2025-2028 document to answer. "
            f"Do NOT make up information not found below. If the answer isn't in the excerpts, "
            f"say so politely and suggest they contact the DTS office.\n\n"
            f"--- Document Excerpts ---\n{rag_context}\n--- End of Excerpts ---\n\n"
            f"Provide a clear, concise, and helpful answer based on the above."
        )

    return None
