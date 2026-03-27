-- MySQL: metadata ONLY
-- No vectors stored here

CREATE TABLE IF NOT EXISTS documents (
    chunk_id    VARCHAR(36)  PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL,
    text        TEXT         NOT NULL,
    source      VARCHAR(512) NOT NULL,
    page_number INT          NULL,
    created_at  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_document_id (document_id),
    INDEX idx_source (source)
);
