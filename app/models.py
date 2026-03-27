from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Ingest (direct text chunks) ─────────────────────────────────────


class ChunkInput(BaseModel):
    """A single text chunk to ingest."""
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    document_id: Optional[str] = None
    page_number: Optional[int] = None


class IngestChunksRequest(BaseModel):
    """Request body for POST /ingest/chunks."""
    chunks: list[ChunkInput] = Field(..., min_length=1)


class IngestedChunk(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    page_number: Optional[int] = None


class IngestResponse(BaseModel):
    message: str
    count: int
    chunks: list[IngestedChunk]


# ── Legacy / File Ingest ──────────────────────────────────────────

class StrategyResult(BaseModel):
    strategy_name: str
    chunk_count: int
    embedding_model: str
    vector_size: int
    overwritten: bool

class FileIngestResponse(BaseModel):
    kb_name: str
    source_filename: str
    strategies_processed: list[StrategyResult]
    ingested_at: str


# ── Retrieve ────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: Optional[int] = Field(default=None, ge=1)


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    source: str
    document_id: str
    page_number: Optional[int] = None
    similarity_score: float
    created_at: str


class SearchResponse(BaseModel):
    query: str
    k_used: int
    results: list[SearchResult]


# ── Stats ───────────────────────────────────────────────────────────


class StatsResponse(BaseModel):
    pg_embeddings: int
    mysql_documents: int
    flat_file_sources: int


# ── Errors ──────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error: str
    message: str
