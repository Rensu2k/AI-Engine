"""Template-based response generator with response variation."""

import random
from typing import Dict, Any, Optional, List


# Response templates per intent for natural variation
_GREETING_RESPONSES = [
    "Hello! 👋 I'm the DTS AI Assistant — built by Clarence Buenaflor, Jester Pastor & Mharjade Enario. I can help you check the status of your documents. Just ask me about your document or provide a Tracking No.!",
    "Hi there! 😊 Welcome to the DTS AI Engine — developed by Clarence Buenaflor, Jester Pastor & Mharjade Enario. How can I help you today? You can ask me to check your document status or provide a Tracking No.",
    "Hey! 👋 I'm the DTS AI — built by Clarence, Jester & Mharjade — and I'm here to help you track your documents. Just give me your Tracking No. or ask about your document status!",
]

_THANKS_RESPONSES = [
    "You're welcome! 😊 If you need anything else, feel free to ask.",
    "No problem! Happy to help. Let me know if you need to check another document.",
    "Glad I could help! 😊 Don't hesitate to come back if you need anything.",
    "You're welcome! If you have more documents to check, just provide another Tracking No.",
]

_GOODBYE_RESPONSES = [
    "Goodbye! 👋 Have a great day!",
    "See you! 👋 Don't hesitate to come back if you need help with your documents.",
    "Take care! 😊 I'll be here whenever you need to track a document.",
    "Bye! 👋 Have a wonderful day ahead!",
]

_COMPLAINT_RESPONSES = [
    (
        "I'm sorry to hear you're having trouble with your document. "
        "If you'd like, I can check the current status of your document — just provide your PDID number.\n\n"
        "If you wish to formally file a complaint or escalate your concern, "
        "please visit the DTS office or contact the City Administrator's Office for assistance."
    ),
    (
        "I understand your frustration, and I'm sorry for the inconvenience. "
        "I can help by checking the latest status of your document — just give me the Tracking No.\n\n"
        "For formal complaints, please reach out to the City Administrator's Office."
    ),
]

_LGU_RESPONSES = [
    "The City Government of Surigao is committed to providing efficient, transparent, and responsive public service. For specific inquiries about city ordinances, mayor's office programs, or local government services, you can visit the official Surigao City website or the City Hall.",
    "Surigao City, known as the 'City of Island Adventures,' is governed by dedicated local officials focused on sustainable development and public welfare. If you need details on specific LGU programs, I recommend contacting the City Information Office."
]

_TOURISM_RESPONSES = [
    "Surigao City is famous for its beautiful islands and beaches! Some top tourist spots include Mabua Pebble Beach, Day-asan Floating Village, Songkoy Cold Spring, and the various islands like Basul and Silop. It's truly a City of Island Adventures!",
    "Looking for places to visit in Surigao? You shouldn't miss Mabua Pebble Beach for its unique stone shoreline, the mangrove forests of Day-asan Floating Village, or island hopping around the city. Don't forget to try the local seafood!"
]


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
        return random.choice(_GREETING_RESPONSES)

    # --- Help ---
    if intent == "help":
        return (
            "🤖 **DTS AI Engine v1.0** — *Built by Clarence Buenaflor, Jester Pastor & Mharjade Enario*\n\n"
            "Here's how I can help you:\n\n"
            "📄 **Check Document Status** — Ask me something like \"What is the status of my document?\" "
            "and I'll look it up for you.\n\n"
            "🔍 **Track by Tracking No.** — If you know your document's Tracking No., just say \"1000\" "
            "and I'll fetch the latest status.\n\n"
            "Just type your question and I'll do my best to assist you!"
        )

    # --- Complaint ---
    if intent == "complaint":
        return random.choice(_COMPLAINT_RESPONSES)

    # --- Thanks ---
    if intent == "thanks":
        return random.choice(_THANKS_RESPONSES)

    # --- Goodbye ---
    if intent == "goodbye":
        return random.choice(_GOODBYE_RESPONSES)

    # --- LGU Query ---
    if intent == "lgu_query":
        return random.choice(_LGU_RESPONSES)

    # --- Tourism Query ---
    if intent == "tourism_query":
        return random.choice(_TOURISM_RESPONSES)

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
        f"• **Accountable:** {current_holder}\n"
    )

    response += (
        f"• **Title:** {title}\n"
        f"• **Status:** {status_icon} {status}\n"
        f"• **Current Location:** {current_office}\n"
    )

    response += (
        f"• **Current Action:** {current_action}\n"
        f"• **Origin Office:** {origin_office}\n"
        f"• **Created By:** {created_by}\n"
        f"• **Date Created:** {created_at}\n"
        f"• **Overall Days on Process:** {total_time}\n"
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
            tat = stop.get("tat", "N/A")

            # Clean up TAT by stripping trailing spaces or using N/A if empty
            if tat and tat != "N/A":
                tat_str = f" [TAT: {tat.strip()}]"
            else:
                tat_str = ""

            if date_out == "Still here":
                response += f"  {i}. 📍 **{office}** — {action} (held by {holder}){tat_str} ← *Currently here*\n"
            else:
                response += f"  {i}. {office} — {action} (handled by {holder}){tat_str}\n"

    return response
