"""Template-based response generator."""

from typing import Dict, Any, Optional, List


def generate_response(intent: str, entities: Dict[str, str], document: Optional[Dict[str, Any]] = None, context: dict = None) -> str:
    """
    Generate a human-friendly response based on intent, entities, and document data.

    Args:
        intent: Classified intent label
        entities: Extracted entities (e.g., {"pdid": "001"})
        document: Parsed document data from DTS API (or None)
        context: Session context for multi-turn awareness

    Returns:
        Response string to send to the user.
    """

    # --- Document status with data --- 
    if intent == "document_status" and document:
        return _format_document_status(document)

    # --- Document status but no PDID provided ---
    if intent == "document_status" and "pdid" not in entities:
        return "I can help you check your document status. What is the Tracking No. of your document?"

    # --- Document status with PDID but no data found ---
    if intent == "document_status" and "pdid" in entities and document is None:
        pdid = entities["pdid"]
        return f"I couldn't find any document with Tracking No. {pdid}. Please double-check the Tracking No. and try again."

    # --- Follow-up with document data ---
    if intent == "follow_up" and document:
        return _format_document_status(document)

    # --- Follow-up with PDID but no document found ---
    if intent == "follow_up" and "pdid" in entities and document is None:
        pdid = entities["pdid"]
        return f"I couldn't find any document with Tracking No. {pdid}. Please check the Tracking No. and try again."

    # --- Follow-up without PDID ---
    if intent == "follow_up" and "pdid" not in entities:
        return "I didn't catch the Tracking No. Could you please provide the Tracking No. of the document you want to check?"

    # --- Greeting ---
    if intent == "greeting":
        return "Hello! I'm the Document Tracking Assistant. I can help you check the status of your documents. Just ask me about your document or provide a Tracking No.!"

    # --- Help ---
    if intent == "help":
        return (
            "Here's how I can help you:\n\n"
            "📄 **Check Document Status** — Ask me something like \"What is the status of my document?\" "
            "and I'll look it up for you.\n\n"
            "🔍 **Track by Tracking No.** — If you know your document's Tracking No., just say \"1000\" "
            "and I'll fetch the latest status.\n\n"
            "Just type your question and I'll do my best to assist you!"
        )

    # --- Complaint ---
    if intent == "complaint":
        return (
            "I'm sorry to hear you're having trouble with your document. "
            "If you'd like, I can check the current status of your document — just provide your PDID number.\n\n"
            "If you wish to formally file a complaint or escalate your concern, "
            "please visit the DTS office or contact the City Administrator's Office for assistance."
        )

    # --- Unknown / fallback ---
    return (
        "I'm sorry, I didn't understand that. I'm a Document Tracking Assistant — "
        "I can help you check the status of your documents. "
        "Try asking something like \"What is the status of my document?\" or provide a Tracking No."
    )


def _format_document_status(document: Dict[str, Any]) -> str:
    """Format parsed document data into a rich, readable response."""
    pdid = document.get("pdid", "N/A")
    title = document.get("title", "N/A")
    status = document.get("status", "N/A")
    current_office = document.get("current_office", "N/A")
    current_holder = document.get("current_holder", "N/A")
    current_action = document.get("current_action", "N/A")
    origin_office = document.get("origin_office", "N/A")
    created_by = document.get("created_by", "N/A")
    created_at = document.get("created_at", "N/A")
    total_time = document.get("total_time", "N/A")
    route_count = document.get("route_count", 0)
    is_completed = document.get("is_completed", False)

    # Status emoji
    if is_completed:
        status_icon = "✅"
    else:
        status_icon = "🔄"

    response = (
        f"📄 **Document Status for PDID {pdid}**\n\n"
        f"• **Title:** {title}\n"
        f"• **Status:** {status_icon} {status}\n"
        f"• **Current Location:** {current_office}\n"
        f"• **Current Holder:** {current_holder}\n"
        f"• **Current Action:** {current_action}\n"
        f"• **Origin Office:** {origin_office}\n"
        f"• **Created By:** {created_by}\n"
        f"• **Date Created:** {created_at}\n"
        f"• **Total Processing Time:** {total_time}\n"
        f"• **Offices Visited:** {route_count}\n"
    )

    # Add route history
    route_summary = document.get("route_summary", [])
    if route_summary:
        response += "\n📋 **Route History:**\n"
        for i, stop in enumerate(route_summary, 1):
            office = stop.get("office", "Unknown")
            holder = stop.get("holder", "N/A")
            action = stop.get("action", "N/A")
            date_out = stop.get("date_out", "")

            if date_out == "Still here":
                response += f"  {i}. 📍 **{office}** — {action} (held by {holder}) ← *Currently here*\n"
            else:
                response += f"  {i}. {office} — {action} (handled by {holder})\n"

    return response
