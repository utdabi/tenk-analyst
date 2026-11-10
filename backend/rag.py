"""Shared RAG code: the Qdrant connection, the OpenRouter embeddings, and search filters.

Ingest, the pipelines, and the agent all import from this file.
"""
import functools
import os
import tempfile

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from qdrant_client import QdrantClient, models

load_dotenv()  # reads the .env file into environment variables

OPENROUTER_URL = "https://openrouter.ai/api/v1"
COLLECTION = os.environ.get("QDRANT_COLLECTION", "tenk")

# The ten companies: ticker -> name used in messages and prompts.
COMPANIES = {
    "TSLA": "Tesla",
    "GM": "General Motors",
    "F": "Ford",
    "RIVN": "Rivian",
    "LCID": "Lucid",
    "APTV": "Aptiv",
    "BWA": "BorgWarner",
    "ALV": "Autoliv",
    "HOG": "Harley-Davidson",
    "CVNA": "Carvana",
}
YEARS = [2024, 2025]


# @functools.cache runs the function once and remembers the result,
# so the whole app shares one connection.
@functools.cache
def get_client() -> QdrantClient:
    return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=60)


@functools.cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",  # 1536 numbers per vector
        base_url=OPENROUTER_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        check_embedding_ctx_length=False,  # required for OpenRouter
    )


@functools.cache
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("CHAT_MODEL", "anthropic/claude-haiku-5.5"),
        base_url=OPENROUTER_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
        max_tokens=1500,
    )


def ensure_collection() -> None:
    """Create the collection and its indexes if they do not exist yet. Safe to run again."""
    client = get_client()
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))
    # An index on each field that we filter on makes filtered search fast.
    # LangChain stores metadata under "metadata.", so the keys start with it.
    client.create_payload_index(COLLECTION, "metadata.company", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.item", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "metadata.year", models.PayloadSchemaType.INTEGER)


@functools.cache
def get_store() -> QdrantVectorStore:
    return QdrantVectorStore(client=get_client(), collection_name=COLLECTION, embedding=get_embeddings())


def make_filter(companies=None, years=None, items=None):
    """Build a Qdrant filter. An empty or missing list means "do not filter on this"."""
    must = []
    if companies:
        must.append(models.FieldCondition(key="metadata.company", match=models.MatchAny(any=companies)))
    if years:
        must.append(models.FieldCondition(key="metadata.year", match=models.MatchAny(any=years)))
    if items:
        must.append(models.FieldCondition(key="metadata.item", match=models.MatchAny(any=items)))
    return models.Filter(must=must) if must else None


# @traceable makes LangSmith show this function as its own step, with its input, output, and time.
@traceable(run_type="retriever", name="qdrant_search")
def search(question: str, k: int = 5, flt=None) -> list:
    """Return the k chunks whose meaning is closest to the question."""
    return get_store().similarity_search(question, k=k, filter=flt)


@functools.cache
def get_ranker() -> Ranker:
    """The FlashRank model (about 3 MB, runs on the CPU). It is downloaded the first time."""
    return Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=os.path.join(tempfile.gettempdir(), "flashrank"))


@traceable(name="flashrank_rerank")
def rerank(question: str, docs: list, top_n: int = 5) -> list:
    """Re-sort the chunks by how well they fit the question and keep the best top_n."""
    if not docs:
        return []
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    ranked = get_ranker().rerank(RerankRequest(query=question, passages=passages))  # best first
    return [docs[r["id"]] for r in ranked[:top_n]]


def doc_to_source(doc) -> dict:
    """Turn a chunk into the source dict that the API returns."""
    meta = doc.metadata
    return {"type": "10-K", "company": meta["company"], "year": meta["year"], "item": meta["item"],
            "item_title": meta["item_title"], "excerpt": doc.page_content[:300], "url": meta["source"]}


def count_points(companies=None, years=None) -> int:
    """Count the chunks in Qdrant, optionally only for some companies and years."""
    return get_client().count(COLLECTION, count_filter=make_filter(companies, years), exact=True).count
