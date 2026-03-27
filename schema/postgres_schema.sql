-- PostgreSQL with pgvector: embeddings ONLY
-- No metadata stored here

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id   VARCHAR(36) PRIMARY KEY,
    embedding  vector(384) NOT NULL
);

-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_embedding_hnsw
ON embeddings USING hnsw (embedding vector_cosine_ops);
