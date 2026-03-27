from __future__ import annotations

import logging

from fastapi import APIRouter

from app.errors import error_response
from app.models import IngestChunksRequest
from app.services.ingestion import ingest_chunks

router = APIRouter()
logger = logging.getLogger("nexvec.ingest")


@router.post("/ingest/chunks")
def ingest(request: IngestChunksRequest):
    """
    Ingest text chunks:
    → Generate embeddings (sentence-transformers)
    → Insert metadata into MySQL
    → Insert embeddings into PostgreSQL pgvector
    → Store chunk_id + embeddings into flat file
    """
    try:
        logger.info("Ingest request: %d chunks", len(request.chunks))
        response = ingest_chunks(request.chunks)
        return response.model_dump()
    except ValueError as exc:
        return error_response(400, "VALIDATION_ERROR", str(exc))
    except RuntimeError as exc:
        return error_response(500, "INGESTION_ERROR", str(exc))
    except Exception as exc:
        logger.exception("Unexpected ingestion error")
        return error_response(500, "INGESTION_ERROR", str(exc))
