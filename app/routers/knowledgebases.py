from __future__ import annotations

import logging

from fastapi import APIRouter

from app.services.store import pg_vector_store, mysql_store
from app.services.store.flat_file_store import FlatFileStore

router = APIRouter()
logger = logging.getLogger("nexvec.stats")


@router.get("/stats")
def stats():
    """System stats across all three storage tiers."""
    flat_store = FlatFileStore()
    return {
        "pg_embeddings": pg_vector_store.count_embeddings(),
        "mysql_documents": mysql_store.count_documents(),
        "flat_file_sources": len(flat_store.list_sources()),
    }


@router.get("/sources")
def list_sources():
    """List all distinct document sources from MySQL."""
    sources = mysql_store.list_sources()
    return {"sources": sources}
