from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import get_config

logger = logging.getLogger("nexvec.flat_file")


class FlatFileStore:
    """
    JSONL flat file store for chunk_id + embedding vectors.
    One file per source (partitioned by source filename).
    Each line: {"chunk_id": "uuid", "embedding": [float, ...]}
    """

    def __init__(self) -> None:
        self.config = get_config()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.config.vector_store_path.mkdir(parents=True, exist_ok=True)

    def _file_path(self, source: str) -> Path:
        safe_name = source.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return self.config.vector_store_path / f"{safe_name}.jsonl"

    def write_batch(self, source: str, rows: list[dict[str, Any]]) -> None:
        """
        Append embedding rows to the flat file for the given source.
        Each row: {"chunk_id": str, "embedding": list[float]}
        """
        if not rows:
            return
        path = self._file_path(source)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                line = json.dumps(
                    {"chunk_id": row["chunk_id"], "embedding": row["embedding"]},
                    ensure_ascii=False,
                )
                f.write(line + "\n")
        logger.info("Wrote %d vectors to flat file: %s", len(rows), path.name)

    def read_all(self, source: str) -> list[dict[str, Any]]:
        """Read all embedding rows from a source's flat file."""
        path = self._file_path(source)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def read_by_ids(self, source: str, chunk_ids: set[str]) -> dict[str, list[float]]:
        """Read embeddings for specific chunk_ids. Returns {chunk_id: embedding}."""
        result: dict[str, list[float]] = {}
        for row in self.read_all(source):
            if row["chunk_id"] in chunk_ids:
                result[row["chunk_id"]] = row["embedding"]
        return result

    def list_sources(self) -> list[str]:
        """List all source files in the flat file store."""
        self._ensure_dir()
        return sorted(p.stem for p in self.config.vector_store_path.glob("*.jsonl"))
