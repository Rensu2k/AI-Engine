"""Template-based response generator."""

from typing import Dict, Any, Optional


def generate_response(intent: str, entities: Dict[str, str], document: Optional[Dict[str, Any]] = None, context: dict = None) -> str:
    """
    Generate a human-friendly response based on intent, entities, and document data.

    Args:
        intent: Classified intent label
        entities: Extracted entities (e.g., {"pdid": "001"})
        document: Document data from DTS API (or None)
        context: Session context for multi-turn awareness

    Returns:
        Response string to send to the user.
    """

    # --- Document status with data ---
    if intent == "document_status" and document:
        return _format_document_status(document)

    # --- Document status but no PDID provided ---
    if intent == "document_status" and "pdid" not in entities:
        return "I can help you check your document status. What is the PDID of your document?"

    # --- Document status with PDID but no data found ---
    if intent == "document_status" and "pdid" in entities and document is None:
        pdid = entities["pdid"]
        return f"I couldn't find any document with PDID {pdid}. Please double-check the PDID and try again."

    # --- Follow-up with document data ---
    if intent == "follow_up" and document:
        return _format_document_status(document)

    # --- Follow-up with PDID but no document found ---
    if intent == "follow_up" and "pdid" in entities and document is None:
        pdid = entities["pdid"]
        return f"I couldn't find any document with PDID {pdid}. Please check the PDID and try again."

    # --- Follow-up without PDID ---
    if intent == "follow_up" and "pdid" not in entities:
        return "I didn't catch the PDID. Could you please provide the PDID of the document you want to check?"

    # --- Greeting ---
    if intent == "greeting":
        return "Hello! I'm the Document Tracking Assistant. I can help you check the status of your documents. Just ask me about your document or provide a PDID!"

    # --- Help ---
    if intent == "help":
        return (
            "Here's how I can help you:\n\n"
            "📄 **Check Document Status** — Ask me something like \"What is the status of my document?\" "
            "and I'll look it up for you.\n\n"
            "🔍 **Track by PDID** — If you know your document's PDID, just say \"PDID 001\" "
            "and I'll fetch the latest status.\n\n"
            "Just type your question and I'll do my best to assist you!"
        )

    # --- Unknown / fallback ---
    return (
        "I'm sorry, I didn't understand that. I'm a Document Tracking Assistant — "
        "I can help you check the status of your documents. "
        "Try asking something like \"What is the status of my document?\" or provide a PDID number."
    )


def _format_document_status(document: Dict[str, Any]) -> str:
    """Format document data into a readable response."""
    pdid = document.get("pdid", "N/A")
    title = document.get("title", "N/A")
    status = document.get("status", "N/A")
    department = document.get("current_department", "N/A")
    submitted_by = document.get("submitted_by", "N/A")
    submitted_date = document.get("submitted_date", "N/A")
    remarks = document.get("remarks", "")

    response = (
        f"📄 **Document Status for PDID {pdid}**\n\n"
        f"• **Title:** {title}\n"
        f"• **Status:** {status}\n"
        f"• **Current Department:** {department}\n"
        f"• **Submitted By:** {submitted_by}\n"
        f"• **Date Submitted:** {submitted_date}\n"
    )

    if remarks:
        response += f"• **Remarks:** {remarks}\n"

    return response
