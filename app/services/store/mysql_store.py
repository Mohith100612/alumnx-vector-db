from __future__ import annotations

import logging
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from app.config import get_config

logger = logging.getLogger("nexvec.mysql_store")


def get_connection() -> pymysql.Connection:
    """Create a new MySQL connection."""
    config = get_config()
    try:
        conn = pymysql.connect(
            host=config.db_host,
            port=config.db_port,
            user=config.db_user,
            password=config.db_password,
            database=config.db_name,
            cursorclass=DictCursor,
            autocommit=False,
        )
        return conn
    except pymysql.MySQLError as exc:
        logger.error("MySQL connection failed: %s", exc)
        raise RuntimeError(f"MySQL connection failed: {exc}") from exc


def init_table() -> None:
    """Create the database and documents metadata table if they don't exist."""
    config = get_config()
    # 1. First connect without selecting a database to ensure it exists
    conn_setup = pymysql.connect(
        host=config.db_host,
        port=config.db_port,
        user=config.db_user,
        password=config.db_password,
        autocommit=True,
    )
    with conn_setup.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{config.db_name}`")
    conn_setup.close()

    # 2. Now connect normally with the database selected
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    chunk_id    VARCHAR(36)  PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    text        TEXT         NOT NULL,
                    source      VARCHAR(512) NOT NULL,
                    page_number INT          NULL,
                    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_document_id (document_id),
                    INDEX idx_source (source)
                )
            """)
        conn.commit()
        logger.info("MySQL table 'documents' is ready")
    except pymysql.MySQLError as exc:
        conn.rollback()
        logger.error("Failed to create MySQL table: %s", exc)
        raise
    finally:
        conn.close()


def batch_insert(rows: list[dict[str, Any]]) -> int:
    """
    Batch insert metadata rows into MySQL.
    Each row: {"chunk_id", "document_id", "text", "source", "page_number", "created_at"}
    Uses INSERT IGNORE to skip duplicate chunk_ids.
    Returns number of rows inserted.
    """
    if not rows:
        return 0

    config = get_config()
    conn = get_connection()
    total_inserted = 0
    try:
        with conn.cursor() as cur:
            sql = """
                INSERT IGNORE INTO documents
                    (chunk_id, document_id, text, source, page_number, created_at)
                VALUES
                    (%(chunk_id)s, %(document_id)s, %(text)s, %(source)s,
                     %(page_number)s, %(created_at)s)
            """
            for start in range(0, len(rows), config.batch_size):
                batch = rows[start : start + config.batch_size]
                cur.executemany(sql, batch)
                total_inserted += len(batch)
                logger.debug("Inserted batch %d–%d into MySQL", start, start + len(batch))
        conn.commit()
        logger.info("Inserted %d metadata rows into MySQL", total_inserted)
        return total_inserted
    except pymysql.MySQLError as exc:
        conn.rollback()
        logger.error("Failed to insert metadata: %s", exc)
        raise
    finally:
        conn.close()


def fetch_by_ids(chunk_ids: list[str]) -> list[dict]:
    """
    Fetch metadata from MySQL for the given chunk_ids.
    Preserves the order of input chunk_ids.
    """
    if not chunk_ids:
        return []

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(chunk_ids))
            cur.execute(
                f"SELECT chunk_id, document_id, text, source, page_number, created_at "
                f"FROM documents WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            rows = cur.fetchall()
            # Convert datetime to ISO string
            for row in rows:
                if row.get("created_at"):
                    row["created_at"] = row["created_at"].isoformat()
            # Preserve input order
            row_map = {row["chunk_id"]: row for row in rows}
            return [row_map[cid] for cid in chunk_ids if cid in row_map]
    finally:
        conn.close()


def list_sources() -> list[str]:
    """Return distinct source names from the documents table."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT source FROM documents ORDER BY source")
            return [row["source"] for row in cur.fetchall()]
    finally:
        conn.close()


def list_document_ids() -> list[str]:
    """Return distinct document_ids."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT document_id FROM documents ORDER BY document_id")
            return [row["document_id"] for row in cur.fetchall()]
    finally:
        conn.close()


def count_documents() -> int:
    """Return total number of metadata rows."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM documents")
            return cur.fetchone()["cnt"]
    finally:
        conn.close()


def delete_by_ids(chunk_ids: list[str]) -> int:
    """Delete metadata by chunk_ids. Returns affected row count."""
    if not chunk_ids:
        return 0
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(chunk_ids))
            cur.execute(f"DELETE FROM documents WHERE chunk_id IN ({placeholders})", chunk_ids)
            affected = cur.rowcount
        conn.commit()
        return affected
    finally:
        conn.close()
