"""The FastAPI back end.

Start it with:  uv run python -m uvicorn backend.api:app
"""
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend import agent, pipelines

# Without this, Python shows only warnings and errors, so our info lines would not appear in the Render logs.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # keep the server log short; it logs every request otherwise
log = logging.getLogger("tenk")

REQUIRED = ["OPENROUTER_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "NEON_DATABASE_URL", "SEC_USER_AGENT",
            "ACCESS_CODE"]
PIPELINES = ["simple", "filters", "rerank", "decompose", "agent"]
MAX_QUESTION_LENGTH = 1000

UNAVAILABLE = "The answer service is not available right now. Try again in a minute."


def check_environment() -> None:
    """Stop at start-up if a variable is missing or has a line break. Print the name, never the value."""
    for name in REQUIRED:
        if not os.environ.get(name):
            raise SystemExit(f"Missing environment variable: {name}")
    # A pasted key with an extra line fails later, and the library's error message would print the key.
    for name in REQUIRED + ["LANGSMITH_API_KEY"]:  # LangSmith is optional, but must not have a line break
        value = os.environ.get(name, "")
        if "\n" in value or "\r" in value:
            raise SystemExit(f"The value of {name} contains a line break. Fix it in .env or in the Render dashboard.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.time()
    log.info("Starting: checking the environment variables")
    check_environment()
    log.info("Starting: opening the Neon connections")
    agent.get_checkpointer()  # opens the Neon connections now, so a problem shows at start-up
    log.info("Ready after %.1f s", time.time() - started)  # this is the cold-start time of the back end
    yield
    agent.close_memory()


app = FastAPI(title="10-K Analyst", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    chat_id: str
    pipeline: str
    companies: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}  # calls no paid service


# A plain "def" (not "async def") makes FastAPI run it in a worker thread, so a slow answer does not block /health.
@app.post("/ask")
def ask(request: AskRequest, x_access_code: str | None = Header(default=None)):
    # The checks run in the order of the error table.
    expected = os.environ.get("ACCESS_CODE", "")
    if not expected or not x_access_code or not secrets.compare_digest(x_access_code, expected):
        log.warning("/ask refused: wrong or missing access code")  # never log the codes themselves
        return error(401, "Wrong or missing access code.")
    if not request.question.strip():
        return error(400, "The question is empty.")
    if len(request.question) > MAX_QUESTION_LENGTH:
        return error(400, f"The question is too long. Use at most {MAX_QUESTION_LENGTH:,} characters.")
    if request.pipeline not in PIPELINES:
        return error(400, "Unknown pipeline. Use simple, filters, rerank, decompose, or agent.")

    started = time.time()
    # Log the pipeline and the length, not the question: questions can contain personal data.
    log.info("/ask received: pipeline=%s, question length=%d", request.pipeline, len(request.question))
    try:
        result = pipelines.answer(request.question, request.pipeline, request.companies, request.years,
                                  request.chat_id)
        log.info("/ask answered in %.1f s", time.time() - started)
        return result
    except Exception as e:  # OpenRouter, Qdrant, Neon, or SEC failed
        # Log only the type of the error. Its message can contain a key. Never return a stack trace.
        log.error("The answer failed after %.1f s: %s", time.time() - started, type(e).__name__)
        return error(503, UNAVAILABLE)
