from __future__ import annotations

import logging
import uuid
from pathlib import Path

from app.config import get_config
from app.models import FileIngestResponse, StrategyResult
from app.services.chunking.registry import get_chunker_registry
from app.services.embedding.embedder import GeminiEmbedder
from app.services.pdf_extractor import extract_pdf_pages
from app.services.store import pg_vector_store, mysql_store
from app.services.store.faiss_store import FaissStore
from app.utils import now_ist_iso, slugify_name


SUPPORTED_CHUNKING_STRATEGIES = {"fixed_length", "paragraph", "both"}
logger = logging.getLogger("nexvec.ingest_file")


def _resolve_kb_name(source_filename: str, provided_kb_name: str | None) -> str:
    if provided_kb_name:
        return slugify_name(provided_kb_name)
    stem = slugify_name(source_filename)
    return stem


def _chunk_page_text(chunker, page_number: int, text: str) -> list[tuple[int, str]]:
    chunks = chunker.split(text)
    return [(page_number, chunk) for chunk in chunks]


def ingest_file(
    file_name: str,
    file_path: str,
    kb_name: str | None,
    chunking_strategy: str,
    chunk_size: int | None,
    overlap_size: int | None,
    embedding_model: str | None,
    overwrite: bool,
) -> FileIngestResponse:
    config = get_config()
    resolved_kb_name = _resolve_kb_name(file_name, kb_name)
    logger.info("Resolved kb_name=%s for source=%s", resolved_kb_name, file_name)

    if chunking_strategy not in SUPPORTED_CHUNKING_STRATEGIES:
        raise ValueError("Unsupported chunking strategy requested.")

    strategy_names = ["fixed_length", "paragraph"] if chunking_strategy == "both" else [chunking_strategy]

    # Due to full architectural rewrite to 3-tier, duplicate checking is simplified 
    # to let MySQL handle unique chunk_ids or overwrite based on document_id/source.
    # We will delete old entries if overwrite is true.
    if overwrite:
        logger.info("Overwrite requested. Currently skipped for 3-tier but could delete by source.")

    ext = Path(file_path).suffix.lower()
    is_media = ext in (".png", ".jpg", ".jpeg", ".mp4", ".mov", ".mp3", ".wav", ".m4a")

    created_at = now_ist_iso()

    # We collect all rows to push into the 3-tier architecture
    mysql_rows = []
    pg_rows = []
    faiss_rows = []

    strategies_processed: list[StrategyResult] = []

    # Use GeminiEmbedder for ALL file types (text + media)
    embedder = GeminiEmbedder()

    if is_media:
        
        logger.info("Generating native multimodal embedding for %s", file_name)
        vector = embedder.embed_file(file_path)
        strategy_name = strategy_names[0]
        cid = str(uuid.uuid4())

        mysql_rows.append({
            "chunk_id": cid,
            "document_id": resolved_kb_name,
            "text": f"[media:{file_name}]",
            "source": file_name,
            "page_number": None,
            "created_at": created_at,
        })
        pg_rows.append({"chunk_id": cid, "embedding": vector})
        faiss_rows.append({"chunk_id": cid, "embedding": vector})

        strategies_processed.append(
            StrategyResult(
                strategy_name=strategy_name,
                chunk_count=1,
                embedding_model=embedder.model,
                vector_size=len(vector),
                overwritten=overwrite,
            )
        )
    else:
        # ── PDF / text path: extract → chunk → embed text ─────────────────
        pages = extract_pdf_pages(file_path)
        if not pages:
            raise LookupError("NO_EXTRACTABLE_TEXT")
        logger.info("Extracted %s text pages from %s", len(pages), file_name)

        effective_chunk_size = chunk_size or config.chunk_size
        effective_overlap_size = overlap_size if overlap_size is not None else config.overlap_size
        if effective_overlap_size >= effective_chunk_size:
            raise ValueError("overlap_size must be smaller than chunk_size")

        chunkers = get_chunker_registry(effective_chunk_size, effective_overlap_size)

        for strategy_name in strategy_names:
            logger.info("Starting chunking strategy=%s", strategy_name)
            chunker = chunkers[strategy_name]
            indexed_chunks: list[tuple[int, str]] = []
            for page in pages:
                indexed_chunks.extend(_chunk_page_text(chunker, page.page_number, page.text))

            chunk_texts = [chunk_text for _, chunk_text in indexed_chunks]
            logger.info("Generating embeddings strategy=%s chunk_count=%s", strategy_name, len(chunk_texts))
            vectors = embedder.embed_texts(chunk_texts)
            if len(vectors) != len(indexed_chunks):
                raise RuntimeError("Embedding vector count does not match chunk count.")
            
            chunk_count = 0
            for index, ((page_number, chunk_text), vector) in enumerate(zip(indexed_chunks, vectors)):
                cid = str(uuid.uuid4())
                mysql_rows.append({
                    "chunk_id": cid,
                    "document_id": resolved_kb_name,
                    "text": f"[{strategy_name}] {chunk_text}",
                    "source": file_name,
                    "page_number": page_number,
                    "created_at": created_at,
                })
                pg_rows.append({"chunk_id": cid, "embedding": vector})
                faiss_rows.append({"chunk_id": cid, "embedding": vector})
                chunk_count += 1

            strategies_processed.append(
                StrategyResult(
                    strategy_name=strategy_name,
                    chunk_count=chunk_count,
                    embedding_model=embedder.model,
                    vector_size=config.vector_dimension,
                    overwritten=overwrite,
                )
            )

    # Execute batch insertions across 3-tier architecture
    mysql_store.batch_insert(mysql_rows)
    pg_vector_store.batch_insert(pg_rows)
    
    faiss_store = FaissStore()
    faiss_store.write_batch(file_name, faiss_rows)

    logger.info("Wrote %s rows to 3-tier storage kb_name=%s", len(mysql_rows), resolved_kb_name)
    return FileIngestResponse(
        kb_name=resolved_kb_name,
        source_filename=file_name,
        strategies_processed=strategies_processed,
        ingested_at=created_at,
    )
