# Automotive 10-K RAG Agent

Ask questions about the 10-K filings of ten US auto companies and get answers with sources.

The app searches the text of the filings and reads exact numbers from SEC financial data. Every answer cites where it came from. If the sources do not contain the answer, the app says so. It does not invent numbers.

- **Companies:** Tesla (TSLA), General Motors (GM), Ford (F), Rivian (RIVN), Lucid (LCID), Aptiv (APTV), BorgWarner (BWA), Autoliv (ALV), Harley-Davidson (HOG), Carvana (CVNA)
- **Years:** fiscal years 2024 and 2025 (20 filings)
- **Example questions:**
  - What risks does Tesla report about battery supply?
  - Compare GM and Ford R&D in 2025
  - How did Carvana's revenue change from 2024 to 2025?

Numbers are shown in billions of US dollars, for example "$9.4 billion (FY2025)". A loss has a minus sign: "-$3.7 billion (FY2025)".

## How it works

RAG means *retrieval-augmented generation*: the app first finds the passages of the filings that match the question, then asks a language model to answer using only those passages.

The app is built in stages. Each stage stays available as a "pipeline", so you can compare their answers.

| Pipeline | What it does |
|---|---|
| `simple` | Searches all filings for the 5 best chunks and answers |
| `filters` | Searches only the companies and years named in the question |
| `rerank` | Retrieves 20 chunks, re-scores them with FlashRank, keeps the best 5 |
| `decompose` | Splits a question into up to 4 sub-questions, searches each, then answers from all results |
| `agent` (default) | A LangGraph agent with two tools: `search_10k` (filing text) and `sec_financial_data` (exact numbers: revenue, net income, R&D expense, operating income). It also remembers the conversation. |

Before every pipeline, a **scope check** reads the question. If it names a company outside the ten, or a year other than 2024 or 2025, the app returns a fixed message and searches nothing.

## Stack

| Part | Tool |
|---|---|
| Orchestration | LangGraph and LangChain |
| Vector database | Qdrant Cloud |
| Chat model and embeddings | OpenRouter (embeddings: `openai/text-embedding-3-small`) |
| Reranking | FlashRank |
| Conversation memory | Neon Postgres |
| Financial numbers | SEC XBRL company facts |
| Tracing | LangSmith |
| Back end | FastAPI |
| Front end | Streamlit |
| Hosting | Render (free plan) |

## Project layout

```text
backend/ingest.py        reads the filings and stores the chunks in Qdrant
backend/rag.py           search and rerank (traced in LangSmith)
backend/sec_data.py      reads numbers from SEC financial data
backend/pipelines.py     scope check, answer generation, the four pipelines without tools
backend/agent.py         the agent, with tools and memory
backend/api.py           the FastAPI back end (/ask and /health)
frontend/app.py          the Streamlit front end
data/                    the 20 filings (data/10k) and data/filings.csv
tests/                   one test file per stage
test_data/questions.json the test questions
render.yaml              the two Render services
check_setup.py           checks your accounts and keys before you start
requirements.md          the full specification
plan.md                  the build plan and design decisions
```

## Setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

1. Install the packages:

   ```bash
   uv sync
   ```

2. Copy `.env.example` to `.env` and fill in your own values. Never commit `.env`.

   | Variable | What it is |
   |---|---|
   | `OPENROUTER_API_KEY` | LLM and embeddings |
   | `QDRANT_URL`, `QDRANT_API_KEY` | your Qdrant cluster |
   | `QDRANT_COLLECTION` | collection name (default `tenk`) |
   | `NEON_DATABASE_URL` | direct Neon connection string (host without `-pooler`) |
   | `SEC_USER_AGENT` | your name and email, as SEC requires |
   | `ACCESS_CODE` | code users type before using the app |
   | `API_URL` | where the front end finds the back end (default `http://localhost:8000`) |
   | `CHAT_MODEL` | chat model on OpenRouter; use one your account can access |
   | `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | optional tracing |

3. Check the setup (it never prints a key):

   ```bash
   uv run python check_setup.py
   ```

4. Load the filings into Qdrant. This embeds about 10,000 chunks and costs a few cents on the first run. A second run skips filings that are already stored.

   ```bash
   uv run python -m backend.ingest
   ```

## Run locally

Start the back end in one terminal:

```bash
uv run python -m uvicorn backend.api:app --port 8000
```

Start the front end in a second terminal:

```bash
uv run python -m streamlit run frontend/app.py
```

Open the address Streamlit prints, enter your access code, and ask a question. The sidebar has company and year filters, a "Show sources" switch, "New chat", and a "Learning mode" section for choosing a pipeline.

## The API

`POST /ask` needs the header `X-Access-Code`.

```json
{
  "question": "What was Ford's R&D expense in 2025?",
  "chat_id": "3f9a2c1e-6b1d-4f5a-9e7b-1c2d3e4f5a6b",
  "pipeline": "agent",
  "companies": [],
  "years": []
}
```

The response has `answer`, `pipeline`, `chat_id`, `tools_used`, and `sources`. A source is either SEC data (company, year, metric, value) or a 10-K passage (company, year, Item, excerpt, link). Errors return a short `error` message, never a stack trace.

`GET /health` returns `{"status": "ok"}` and calls no paid service.

## Tests

```bash
uv run pytest
```

There is one test file per stage. Most tests call OpenRouter and cost a few cents in total. They check sources, tools used, numbers, and the fixed messages.

## Deploy on Render

`render.yaml` describes two free web services, the back end and the front end. In the Render dashboard choose **New > Blueprint**, pick the GitHub repository, and type the secret values when asked. Both services need the same `ACCESS_CODE`. The front end's `API_URL` is the back end's public URL.

The free plan sleeps when idle. The first request can take about 30 seconds while the service wakes up; the front end waits and retries.

## Limits

- Only the ten companies and fiscal years 2024 and 2025.
- Financial numbers cover four metrics: revenue, net income, R&D expense, operating income.
- Questions are stored in Neon and LangSmith. Do not type personal data into the app.
- This is a class project. It is not investment advice.

## Data sources

The filings come from SEC EDGAR. `data/filings.csv` lists each filing with its SEC link. Financial numbers come from SEC XBRL company facts.
