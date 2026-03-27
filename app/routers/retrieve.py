from __future__ import annotations

import logging

from fastapi import APIRouter

from app.errors import error_response
from app.models import SearchRequest
from app.services.retrieval_service import search

router = APIRouter()
logger = logging.getLogger("nexvec.retrieve")


@router.post("/retrieve")
def retrieve(request: SearchRequest):
    """
    Query pipeline:
    → Embed query (sentence-transformers)
    → Search PostgreSQL pgvector for top K
    → Fetch metadata from MySQL
    → Return merged results
    """
    try:
        logger.info("Search request: query=%r k=%s", request.query, request.k)
        response = search(request)
        return response.model_dump()
    except ValueError as exc:
        message = str(exc)
        if message == "EMPTY_QUERY":
            return error_response(400, "EMPTY_QUERY", "The query cannot be empty.")
        return error_response(400, "VALIDATION_ERROR", message)
    except Exception as exc:
        logger.exception("Unexpected retrieval error")
        return error_response(500, "RETRIEVAL_ERROR", str(exc))
