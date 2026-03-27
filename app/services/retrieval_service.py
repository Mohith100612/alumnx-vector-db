from __future__ import annotations

import logging

from app.config import get_config
from app.models import SearchRequest, SearchResponse, SearchResult
from app.services.embedding.embedder import GeminiEmbedder
from app.services.store import pg_vector_store, mysql_store

logger = logging.getLogger("nexvec.retrieval")


def search(request: SearchRequest) -> SearchResponse:
    """
    Production retrieval pipeline:
    1. Embed query with Gemini Embedding 2
    2. Search PostgreSQL pgvector → top K chunk_ids + similarity scores
    3. Fetch metadata from MySQL WHERE chunk_id IN (...)
    4. Merge and return
    """
    config = get_config()
    k = request.k or config.default_top_k

    if not request.query.strip():
        raise ValueError("EMPTY_QUERY")

    logger.info("Search started: query=%r k=%d", request.query, k)

    # ── 1. Embed query (Gemini Embedding 2) ───────────────────────────
    embedder = GeminiEmbedder()
    query_vector = embedder.embed_query(request.query)

    # ── 2. Search pgvector ────────────────────────────────────────────
    pg_results = pg_vector_store.search_similar(query_vector, k=k)
    if not pg_results:
        logger.info("pgvector returned no results")
        return SearchResponse(query=request.query, k_used=k, results=[])

    chunk_ids = [r["chunk_id"] for r in pg_results]
    score_map = {r["chunk_id"]: r["similarity_score"] for r in pg_results}
    logger.info("pgvector returned %d results: %s", len(chunk_ids), chunk_ids)

    # ── 3. Fetch metadata from MySQL ──────────────────────────────────
    metadata_rows = mysql_store.fetch_by_ids(chunk_ids)
    logger.info("MySQL returned %d rows for %d chunk_ids", len(metadata_rows), len(chunk_ids))

    # ── 4. Merge results ──────────────────────────────────────────────
    results: list[SearchResult] = []
    for row in metadata_rows:
        cid = row["chunk_id"]
        results.append(
            SearchResult(
                chunk_id=cid,
                text=row["text"],
                source=row["source"],
                document_id=row["document_id"],
                page_number=row.get("page_number"),
                similarity_score=score_map.get(cid, 0.0),
                created_at=row.get("created_at", ""),
            )
        )

    # Sort by similarity descending
    results.sort(key=lambda r: r.similarity_score, reverse=True)
    logger.info("Search complete: returned %d results", len(results))

    return SearchResponse(query=request.query, k_used=k, results=results)
