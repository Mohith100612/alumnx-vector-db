from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from app.models import ChunkInput, IngestResponse, IngestedChunk
from app.services.embedding.embedder import GeminiEmbedder
from app.services.store import pg_vector_store, mysql_store
from app.utils import slugify_name

logger = logging.getLogger("nexvec.ingestion")


def _generate_document_id(source: str) -> str:
    """Generate a deterministic document_id from the source filename."""
    return slugify_name(source)


def ingest_chunks(chunks: list[ChunkInput]) -> IngestResponse:
    """
    Production ingestion pipeline:
    1. Assign UUID chunk_ids
    2. Generate embeddings with Gemini Embedding 2
    3. Batch insert metadata → MySQL
    4. Batch insert embeddings → PostgreSQL pgvector
    5. Write chunk_id + embedding → FAISS
    """
    if not chunks:
        raise ValueError("No chunks provided")

    n = len(chunks)
    logger.info("Ingestion started: %d chunks", n)

    # ── 1. Assign UUIDs ───────────────────────────────────────────────
    chunk_ids = [str(uuid.uuid4()) for _ in range(n)]

    # ── 2. Generate embeddings (Gemini Embedding 2) ───────────────────
    embedder = GeminiEmbedder()
    texts = [c.text for c in chunks]
    vectors = embedder.embed_texts(texts)

    if len(vectors) != n:
        raise RuntimeError(f"Embedding count ({len(vectors)}) != chunk count ({n})")
    logger.info("Generated %d embeddings (dim=%d)", n, len(vectors[0]))

    # ── 3. Insert metadata → MySQL ────────────────────────────────────
    now = datetime.now(tz=timezone.utc).isoformat()
    mysql_rows = [
        {
            "chunk_id": cid,
            "document_id": chunk.document_id or _generate_document_id(chunk.source),
            "text": chunk.text,
            "source": chunk.source,
            "page_number": chunk.page_number,
            "created_at": now,
        }
        for cid, chunk in zip(chunk_ids, chunks)
    ]
    mysql_store.batch_insert(mysql_rows)

    # ── 4. Insert embeddings → PostgreSQL pgvector ────────────────────
    pg_rows = [
        {"chunk_id": cid, "embedding": vec}
        for cid, vec in zip(chunk_ids, vectors)
    ]
    pg_vector_store.batch_insert(pg_rows)

    # ── 5. Write to FAISS (Flat File Alternative) ─────────────────────
    from app.services.store.faiss_store import FaissStore
    faiss_store = FaissStore()
    # Group by source for partitioned FAISS indexes
    by_source: dict[str, list[dict]] = {}
    for cid, chunk, vec in zip(chunk_ids, chunks, vectors):
        by_source.setdefault(chunk.source, []).append(
            {"chunk_id": cid, "embedding": vec}
        )
    for source, rows in by_source.items():
        faiss_store.write_batch(source, rows)

    logger.info("Ingestion complete: %d chunks → MySQL + PostgreSQL + FAISS", n)

    ingested = [
        IngestedChunk(
            chunk_id=cid,
            document_id=chunk.document_id or _generate_document_id(chunk.source),
            source=chunk.source,
            page_number=chunk.page_number,
        )
        for cid, chunk in zip(chunk_ids, chunks)
    ]
    return IngestResponse(
        message=f"Successfully ingested {n} chunks",
        count=n,
        chunks=ingested,
    )
