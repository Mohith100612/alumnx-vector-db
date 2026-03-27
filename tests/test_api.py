from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ── Fake Embedder ─────────────────────────────────────────────────────

class FakeEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "test-model"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


# ── In-memory mocks for all three stores ──────────────────────────────

class FakePGVectorStore:
    def __init__(self):
        self.data: dict[str, list[float]] = {}

    def init_table(self):
        pass

    def batch_insert(self, rows):
        for row in rows:
            self.data[row["chunk_id"]] = row["embedding"]
        return len(rows)

    def search_similar(self, query_embedding, k=5):
        # Return all stored chunk_ids with fake similarity scores
        results = []
        for cid in list(self.data.keys())[:k]:
            results.append({"chunk_id": cid, "similarity_score": 0.95})
        return results

    def count_embeddings(self):
        return len(self.data)

    def delete_by_ids(self, chunk_ids):
        for cid in chunk_ids:
            self.data.pop(cid, None)
        return len(chunk_ids)


class FakeMySQLStore:
    def __init__(self):
        self.data: dict[str, dict] = {}

    def init_table(self):
        pass

    def batch_insert(self, rows):
        for row in rows:
            self.data[row["chunk_id"]] = row.copy()
        return len(rows)

    def fetch_by_ids(self, chunk_ids):
        return [self.data[cid] for cid in chunk_ids if cid in self.data]

    def list_sources(self):
        return sorted(set(r["source"] for r in self.data.values()))

    def count_documents(self):
        return len(self.data)

    def delete_by_ids(self, chunk_ids):
        for cid in chunk_ids:
            self.data.pop(cid, None)
        return len(chunk_ids)


class FakeFlatFileStore:
    def __init__(self):
        self.data: dict[str, list[dict]] = {}

    def write_batch(self, source, rows):
        self.data.setdefault(source, []).extend(rows)

    def list_sources(self):
        return sorted(self.data.keys())


# ── Fixture ───────────────────────────────────────────────────────────

@pytest.fixture()
def client(monkeypatch, tmp_path):
    cfg = SimpleNamespace(
        chunk_size=10,
        overlap_size=2,
        default_chunking_strategy="fixed_length",
        max_paragraph_size=20,
        min_page_text_length=1,
        default_top_k=5,
        embedding_model="test-model",
        vector_dimension=3,
        vector_store_path=tmp_path / "vector_store",
        batch_size=100,
        pg_host="localhost", pg_port=5432, pg_name="test", pg_user="test", pg_password="test",
        db_host="localhost", db_port=3306, db_name="test", db_user="test", db_password="test",
    )

    # Mock config
    monkeypatch.setattr("app.services.ingestion.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.retrieval_service.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.embedding.embedder.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.store.flat_file_store.get_config", lambda: cfg)

    # Mock embedder
    monkeypatch.setattr("app.services.ingestion.Embedder", FakeEmbedder)
    monkeypatch.setattr("app.services.retrieval_service.Embedder", FakeEmbedder)

    # Mock all three stores
    fake_pg = FakePGVectorStore()
    fake_mysql = FakeMySQLStore()
    fake_flat = FakeFlatFileStore()

    monkeypatch.setattr("app.services.store.pg_vector_store.init_table", fake_pg.init_table)
    monkeypatch.setattr("app.services.store.pg_vector_store.batch_insert", fake_pg.batch_insert)
    monkeypatch.setattr("app.services.store.pg_vector_store.search_similar", fake_pg.search_similar)
    monkeypatch.setattr("app.services.store.pg_vector_store.count_embeddings", fake_pg.count_embeddings)

    monkeypatch.setattr("app.services.store.mysql_store.init_table", fake_mysql.init_table)
    monkeypatch.setattr("app.services.store.mysql_store.batch_insert", fake_mysql.batch_insert)
    monkeypatch.setattr("app.services.store.mysql_store.fetch_by_ids", fake_mysql.fetch_by_ids)
    monkeypatch.setattr("app.services.store.mysql_store.list_sources", fake_mysql.list_sources)
    monkeypatch.setattr("app.services.store.mysql_store.count_documents", fake_mysql.count_documents)

    monkeypatch.setattr(
        "app.services.store.flat_file_store.FlatFileStore.write_batch",
        lambda self, source, rows: fake_flat.write_batch(source, rows),
    )
    monkeypatch.setattr(
        "app.services.store.flat_file_store.FlatFileStore.list_sources",
        lambda self: fake_flat.list_sources(),
    )

    yield TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────


def test_ingest_chunks_stores_to_all_three_tiers(client):
    response = client.post("/ingest/chunks", json={
        "chunks": [
            {"text": "Alpha beta gamma", "source": "doc1.pdf", "page_number": 1},
            {"text": "Delta epsilon zeta", "source": "doc1.pdf", "page_number": 2},
        ]
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert len(payload["chunks"]) == 2
    assert all(c["chunk_id"] for c in payload["chunks"])
    assert all(c["source"] == "doc1.pdf" for c in payload["chunks"])


def test_retrieve_returns_results_after_ingest(client):
    # Ingest first
    client.post("/ingest/chunks", json={
        "chunks": [
            {"text": "machine learning algorithms", "source": "ml.pdf"},
            {"text": "deep neural networks", "source": "ml.pdf"},
        ]
    })

    # Now search
    response = client.post("/retrieve", json={"query": "neural networks", "k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "neural networks"
    assert payload["k_used"] == 2
    assert len(payload["results"]) > 0
    assert payload["results"][0]["similarity_score"] > 0


def test_retrieve_empty_query_rejected(client):
    response = client.post("/retrieve", json={"query": "   ", "k": 5})
    assert response.status_code == 400
    assert response.json()["error"] == "EMPTY_QUERY"


def test_stats_endpoint(client):
    client.post("/ingest/chunks", json={
        "chunks": [
            {"text": "test chunk one", "source": "test.pdf"},
        ]
    })

    response = client.get("/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["pg_embeddings"] == 1
    assert payload["mysql_documents"] == 1


def test_sources_endpoint(client):
    client.post("/ingest/chunks", json={
        "chunks": [
            {"text": "first doc", "source": "alpha.pdf"},
            {"text": "second doc", "source": "beta.pdf"},
        ]
    })

    response = client.get("/sources")
    assert response.status_code == 200
    sources = response.json()["sources"]
    assert "alpha.pdf" in sources
    assert "beta.pdf" in sources


def test_ingest_generates_document_id_from_source(client):
    response = client.post("/ingest/chunks", json={
        "chunks": [
            {"text": "hello world", "source": "My Report 2026.pdf"},
        ]
    })
    assert response.status_code == 200
    chunk = response.json()["chunks"][0]
    assert chunk["document_id"] == "my_report_2026"


def test_ingest_uses_provided_document_id(client):
    response = client.post("/ingest/chunks", json={
        "chunks": [
            {"text": "hello world", "source": "report.pdf", "document_id": "custom-id-123"},
        ]
    })
    assert response.status_code == 200
    chunk = response.json()["chunks"][0]
    assert chunk["document_id"] == "custom-id-123"
