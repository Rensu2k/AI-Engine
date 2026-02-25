"""Text preprocessing for the ML pipeline."""

import re
import string


def preprocess_text(text: str) -> str:
    """
    Normalize input text for intent classification.

    Steps:
    1. Lowercase
    2. Normalize whitespace
    3. Preserve PDID patterns (e.g., PDID-001, PDID 001)
    4. Remove excessive punctuation but keep basic structure
    """
    if not text or not text.strip():
        return ""

    # Lowercase
    text = text.lower().strip()

    # Normalize whitespace (multiple spaces/tabs to single space)
    text = re.sub(r"\s+", " ", text)

    # Normalize PDID patterns to a consistent format: "pdid 001"
    # Matches: PDID-001, PDID 001, PDID001, pdid-001, etc.
    text = re.sub(r"pdid[\s\-_]*(\d+)", r"pdid \1", text)

    # Remove punctuation except hyphens and question marks (useful for intent)
    allowed = set(string.ascii_lowercase + string.digits + " -?")
    text = "".join(ch if ch in allowed else " " for ch in text)

    # Final whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()

    return text
