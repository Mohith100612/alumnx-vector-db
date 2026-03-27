from __future__ import annotations

import logging
from typing import Any

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from app.config import get_config

logger = logging.getLogger("nexvec.pg_vector")


def _register_vector(conn) -> None:
    """Register pgvector type with the connection."""
    from pgvector.psycopg2 import register_vector
    register_vector(conn)


def get_connection():
    """Create a new PostgreSQL connection with SSL."""
    config = get_config()
    try:
        conn = psycopg2.connect(
            host=config.pg_host,
            port=config.pg_port,
            dbname=config.pg_name,
            user=config.pg_user,
            password=config.pg_password,
            sslmode="require",
        )
        conn.autocommit = False
        return conn
    except psycopg2.Error as exc:
        logger.error("PostgreSQL connection failed: %s", exc)
        raise RuntimeError(f"PostgreSQL connection failed: {exc}") from exc


def init_table() -> None:
    """Create pgvector extension and embeddings table with HNSW index."""
    config = get_config()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id  VARCHAR(36) PRIMARY KEY,
                    embedding vector NOT NULL
                )
            """)
        conn.commit()
        logger.info("PostgreSQL embeddings table ready (dynamic vectors, no index)", config.vector_dimension)
    except psycopg2.Error as exc:
        conn.rollback()
        logger.error("Failed to init PostgreSQL table: %s", exc)
        raise
    finally:
        conn.close()


def batch_insert(rows: list[dict[str, Any]]) -> int:
    """
    Batch insert embeddings into PostgreSQL.
    Each row: {"chunk_id": str, "embedding": list[float]}
    Uses ON CONFLICT DO NOTHING to skip duplicate chunk_ids.
    Returns number of rows inserted.
    """
    if not rows:
        return 0

    config = get_config()
    conn = get_connection()
    total_inserted = 0
    try:
        _register_vector(conn)
        with conn.cursor() as cur:
            for start in range(0, len(rows), config.batch_size):
                batch = rows[start : start + config.batch_size]
                values = [
                    (row["chunk_id"], np.array(row["embedding"], dtype=np.float32))
                    for row in batch
                ]
                execute_values(
                    cur,
                    "INSERT INTO embeddings (chunk_id, embedding) VALUES %s ON CONFLICT (chunk_id) DO NOTHING",
                    values,
                )
                total_inserted += len(batch)
                logger.debug("Inserted batch %d–%d into PostgreSQL", start, start + len(batch))
        conn.commit()
        logger.info("Inserted %d embeddings into PostgreSQL", total_inserted)
        return total_inserted
    except psycopg2.Error as exc:
        conn.rollback()
        logger.error("Failed to insert embeddings: %s", exc)
        raise
    finally:
        conn.close()


def search_similar(query_embedding: list[float], k: int = 5) -> list[dict]:
    """
    Search pgvector for top-K similar embeddings using cosine distance.
    Returns list of {"chunk_id": str, "similarity_score": float} sorted by similarity desc.
    """
    conn = get_connection()
    try:
        _register_vector(conn)
        query_vec = np.array(query_embedding, dtype=np.float32)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, 1 - (embedding <=> %s) AS similarity
                FROM embeddings
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vec, query_vec, k),
            )
            results = [
                {"chunk_id": row[0], "similarity_score": float(row[1])}
                for row in cur.fetchall()
            ]
        logger.info("pgvector search returned %d results", len(results))
        return results
    except psycopg2.Error as exc:
        logger.error("pgvector search failed: %s", exc)
        raise
    finally:
        conn.close()


def count_embeddings() -> int:
    """Return total number of embeddings stored."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings")
            return cur.fetchone()[0]
    finally:
        conn.close()


def delete_by_ids(chunk_ids: list[str]) -> int:
    """Delete embeddings by chunk_ids. Returns affected row count."""
    if not chunk_ids:
        return 0
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(chunk_ids))
            cur.execute(f"DELETE FROM embeddings WHERE chunk_id IN ({placeholders})", chunk_ids)
            affected = cur.rowcount
        conn.commit()
        return affected
    finally:
        conn.close()
