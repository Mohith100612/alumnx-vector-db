import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env explicitly
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app.models import ChunkInput
from app.services.ingestion import ingest_chunks
from app.services.retrieval_service import search, SearchRequest
from app.services.store import pg_vector_store, mysql_store
import traceback


def main():
    print("Initializing test for 3-Tier Storage Pipeline...")
    
    # 1. Init Tables
    print("\n--- 1. Initializing Tables ---")
    try:
        print("Connecting to PostgreSQL and creating embeddings table...")
        pg_vector_store.init_table()
        print("PostgreSQL table created successfully.")
        
        print("Connecting to MySQL and creating documents table...")
        mysql_store.init_table()
        print("MySQL table created successfully.")
    except Exception as e:
        print(f"Failed to initialize tables: {e}")
        traceback.print_exc()
        return

    # 2. Test Ingestion
    print("\n--- 2. Testing Ingestion (3-way write) ---")
    chunks = [
        ChunkInput(text="Vector databases are great for semantic search.", source="test_doc_1", page_number=1),
        ChunkInput(text="PostgreSQL pgvector extension enables ANN search inside Postgres.", source="test_doc_1", page_number=2),
        ChunkInput(text="MySQL is a popular open-source relational database management system.", source="test_doc_2", page_number=1),
        ChunkInput(text="FAISS is a library for efficient similarity search and clustering of dense vectors.", source="test_doc_2", page_number=2),
    ]
    
    try:
        response = ingest_chunks(chunks)
        print(f"Ingestion successful: {response.message}")
        print(f"Ingested chunks: {[c.chunk_id for c in response.chunks]}")
    except Exception as e:
        print(f"Failed to ingest chunks: {e}")
        traceback.print_exc()
        return

    # 3. Test Retrieval
    print("\n--- 3. Testing Retrieval (pgvector + MySQL) ---")
    try:
        req = SearchRequest(query="What database is used for semantic search?", k=2)
        results = search(req)
        print(f"Search successful! Query: '{results.query}'")
        for i, res in enumerate(results.results, 1):
            print(f"  Result {i}: [{res.similarity_score:.4f}] {res.text} (Source: {res.source}, ID: {res.chunk_id})")
    except Exception as e:
        print(f"Failed to search: {e}")
        traceback.print_exc()
        return

    print("\nAll tests passed perfectly! The 3-tier system is fully operational.")

if __name__ == "__main__":
    main()
