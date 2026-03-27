from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import get_config

logger = logging.getLogger("nexvec.faiss_store")


class FaissStore:
    """
    FAISS index for chunk_id + embedding vectors.
    One index per source.
    Stores chunk_ids in a separate JSON file since FAISS only stores integer IDs natively.
    """

    def __init__(self) -> None:
        self.config = get_config()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.config.vector_store_path.mkdir(parents=True, exist_ok=True)

    def _get_paths(self, source: str) -> tuple[Path, Path]:
        safe_name = source.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(".pdf", "")
        index_path = self.config.vector_store_path / f"{safe_name}.index"
        ids_path = self.config.vector_store_path / f"{safe_name}_ids.json"
        return index_path, ids_path

    def write_batch(self, source: str, rows: list[dict[str, Any]]) -> None:
        """
        Append embedding rows to the FAISS index for the given source.
        Each row: {"chunk_id": str, "embedding": list[float]}
        """
        if not rows:
            return
        
        index_path, ids_path = self._get_paths(source)
        
        # Load or create chunk IDs mapping
        chunk_ids = []
        if ids_path.exists():
            with ids_path.open("r", encoding="utf-8") as f:
                chunk_ids = json.load(f)
        
        # Load or create FAISS index
        dim = self.config.vector_dimension
        if index_path.exists():
            index = faiss.read_index(str(index_path))
        else:
            # Using IndexFlatIP for cosine similarity (assuming normalized vectors)
            # or IndexFlatL2 for L2 distance. We'll use IndexFlatL2 as default unless specified
            index = faiss.IndexFlatL2(dim)
            
        # Add new embeddings
        embeddings_array = np.array([row["embedding"] for row in rows], dtype=np.float32)
        index.add(embeddings_array)
        
        # Add new chunk IDs
        new_ids = [row["chunk_id"] for row in rows]
        chunk_ids.extend(new_ids)
        
        # Save both
        faiss.write_index(index, str(index_path))
        with ids_path.open("w", encoding="utf-8") as f:
            json.dump(chunk_ids, f)
            
        logger.info("Wrote %d vectors to FAISS index: %s", len(rows), index_path.name)

    def list_sources(self) -> list[str]:
        """List all source names in the FAISS store."""
        self._ensure_dir()
        return sorted(p.stem.replace("_ids", "") for p in self.config.vector_store_path.glob("*_ids.json"))
