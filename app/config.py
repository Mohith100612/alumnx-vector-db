from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    # Chunking
    chunk_size: int
    overlap_size: int
    default_chunking_strategy: str
    max_paragraph_size: int
    min_page_text_length: int
    # Retrieval
    default_top_k: int
    # Embedding
    embedding_model: str
    vector_dimension: int
    # Storage
    vector_store_path: Path
    batch_size: int
    # PostgreSQL RDS (pgvector — embeddings only)
    pg_host: str
    pg_port: int
    pg_name: str
    pg_user: str
    pg_password: str
    # MySQL (metadata only)
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str


_CONFIG_CACHE: AppConfig | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_raw_config() -> dict[str, Any]:
    config_path = project_root() / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_config() -> AppConfig:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    raw = _load_raw_config()
    vector_store_path = Path(raw.get("vector_store_path", "./vector_store/"))
    if not vector_store_path.is_absolute():
        vector_store_path = (project_root() / vector_store_path).resolve()

    _CONFIG_CACHE = AppConfig(
        chunk_size=int(raw["chunk_size"]),
        overlap_size=int(raw["overlap_size"]),
        default_chunking_strategy=str(raw["default_chunking_strategy"]),
        max_paragraph_size=int(raw["max_paragraph_size"]),
        min_page_text_length=int(raw.get("min_page_text_length", 20)),
        default_top_k=int(raw.get("default_top_k", 5)),
        embedding_model=str(raw.get("embedding_model", "all-MiniLM-L6-v2")),
        vector_dimension=int(raw.get("vector_dimension", 384)),
        vector_store_path=vector_store_path,
        batch_size=int(raw.get("batch_size", 100)),
        # PostgreSQL
        pg_host=os.environ.get("PG_HOST", "localhost"),
        pg_port=int(os.environ.get("PG_PORT", "5432")),
        pg_name=os.environ.get("PG_NAME", "postgres"),
        pg_user=os.environ.get("PG_USER", "postgres"),
        pg_password=os.environ.get("PG_PASSWORD", ""),
        # MySQL
        db_host=os.environ.get("DB_HOST", "localhost"),
        db_port=int(os.environ.get("DB_PORT", "3306")),
        db_name=os.environ.get("DB_NAME", "alumnx_metadata"),
        db_user=os.environ.get("DB_USER", "root"),
        db_password=os.environ.get("DB_PASSWORD", ""),
    )
    return _CONFIG_CACHE
