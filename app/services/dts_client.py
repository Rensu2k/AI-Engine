"""
DTS API client for fetching document status.

Supports mock mode (built-in sample data) and live mode (HTTP calls to DTS backend).
"""

import httpx
from typing import Optional, Dict, Any
from app.config import settings


# Mock document data for development/testing
MOCK_DOCUMENTS = {
    "001": {
        "pdid": "001",
        "title": "Leave Application - Juan Dela Cruz",
        "status": "Processing",
        "current_department": "HR",
        "submitted_by": "Juan Dela Cruz",
        "submitted_date": "2026-02-20",
        "remarks": "Under review by HR department",
    },
    "002": {
        "pdid": "002",
        "title": "Purchase Request - Office Supplies",
        "status": "Approved",
        "current_department": "Accounting",
        "submitted_by": "Maria Santos",
        "submitted_date": "2026-02-18",
        "remarks": "Approved and forwarded to Accounting for budget allocation",
    },
    "003": {
        "pdid": "003",
        "title": "Travel Order - Conference Attendance",
        "status": "Pending",
        "current_department": "Admin",
        "submitted_by": "Pedro Reyes",
        "submitted_date": "2026-02-22",
        "remarks": "Awaiting Admin officer signature",
    },
    "004": {
        "pdid": "004",
        "title": "Disbursement Voucher - Training Expense",
        "status": "Released",
        "current_department": "Cashier",
        "submitted_by": "Ana Garcia",
        "submitted_date": "2026-02-15",
        "remarks": "Check released on 2026-02-23",
    },
    "005": {
        "pdid": "005",
        "title": "Memorandum - Policy Update",
        "status": "Rejected",
        "current_department": "Legal",
        "submitted_by": "Carlos Mendoza",
        "submitted_date": "2026-02-10",
        "remarks": "Returned for revision. Please update Section 3.",
    },
}


async def get_document(pdid: str) -> Optional[Dict[str, Any]]:
    """
    Fetch document info by PDID.

    In mock mode: returns from built-in sample data.
    In live mode: calls DTS backend API.

    Args:
        pdid: The document PDID (e.g., "001")

    Returns:
        Document info dict or None if not found.
    """
    # Normalize PDID
    pdid = pdid.strip().lstrip("0") or "0"
    pdid = pdid.zfill(3)

    if settings.DTS_MOCK_MODE:
        return MOCK_DOCUMENTS.get(pdid)

    # Live mode: call DTS backend
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{settings.DTS_API_BASE_URL}/api/documents/{pdid}"
            response = await client.get(url)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                response.raise_for_status()
    except httpx.HTTPError:
        return None

    return None
