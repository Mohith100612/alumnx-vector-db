import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dotenv import load_dotenv

from app.routers.ingest import router as ingest_router
from app.routers.ingest_file import router as ingest_file_router
from app.routers.retrieve import router as retrieve_router
from app.routers.knowledgebases import router as stats_router
from app.errors import error_response
from app.services.store import pg_vector_store, mysql_store


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("nexvec")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize both database tables."""
    logger.info("Initializing PostgreSQL pgvector table on RDS...")
    pg_vector_store.init_table()
    logger.info("Initializing MySQL metadata table...")
    mysql_store.init_table()
    logger.info("NexVec 3-tier RAG system ready")
    yield


app = FastAPI(
    title="NexVec",
    version="2.0.0",
    description="3-tier RAG: pgvector (embeddings) + MySQL (metadata) + local FAISS (backup)",
    lifespan=lifespan,
)

app.include_router(ingest_router)
app.include_router(ingest_file_router)
app.include_router(retrieve_router)
app.include_router(stats_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response(422, "VALIDATION_ERROR", "Validation error.", {"errors": exc.errors()})
