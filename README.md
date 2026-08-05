# Earnings Intelligence Platform

A hybrid RAG assistant for querying SEC EDGAR filings — 10-Ks, 10-Qs, and earnings call transcripts — in plain English. Built as the capstone project for [DataTalks.Club LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp).

## Problem

Public company filings contain the information investors and analysts need — margins, risk factors, forward guidance, segment performance — but it's buried across hundreds of pages per filing, repeated quarter over quarter, in dense financial language. Finding a specific fact (e.g. "what did NVIDIA say about data center revenue growth in Q3 2023?") means manually searching PDFs or EDGAR's own interface.

This platform lets a user ask a natural-language question and get a grounded answer sourced directly from the underlying filings, using retrieval that combines keyword and semantic search rather than plain "control-F."

**Corpus:** 1,842 chunks from 10-K/10-Q filings and earnings transcripts for AAPL, MSFT, NVDA, GOOGL, and META, spanning 2019–2023, sourced from SEC EDGAR.

## Architecture

```
EDGAR API → Prefect ingestion (parser + chunker) → Embedder → pgvector (Postgres)

Question → LLM query rewrite ─┐
                               ├→ MinSearch (lexical) ─┐
                               └→ pgvector (semantic) ─┼→ RRF fusion → LLM rerank → LLM answer
                                                        ┘
                          FastAPI ⇄ Streamlit UI          query_log / query_feedback → Grafana
```

## Retrieval Flow

1. **Query rewriting** — the raw question is rewritten by a small LLM call into a retrieval-friendly query (expands abbreviations, drops conversational filler) — see [agent/rag.py](agent/rag.py)
2. **Lexical retrieval** — [MinSearch](https://github.com/alexeygrigorev/minsearch) over filing text
3. **Semantic retrieval** — embeddings stored in **pgvector** (Postgres)
4. **Hybrid fusion** — [Reciprocal Rank Fusion (RRF)](https://en.wikipedia.org/wiki/Reciprocal_rank_fusion) combines both ranked lists — see [search/hybrid.py](search/hybrid.py)
5. **Re-ranking** — an LLM pass re-orders the RRF candidate pool for relevance before truncating to the final top-k — see [search/rerank.py](search/rerank.py)
6. Top results are passed as context to the LLM, which generates a grounded, sourced answer — see [agent/rag.py](agent/rag.py)

## Retrieval Evaluation

Evaluated against 261 ground-truth Q&A pairs (hit rate / MRR) — see [eval/evaluate_retrieval.py](eval/evaluate_retrieval.py) and [eval/EVALUATION_RESULTS.md](eval/EVALUATION_RESULTS.md).

| Approach | Hit Rate | MRR |
|---|---|---|
| **Hybrid (RRF)** | **0.3678** | 0.2065 |
| Semantic only | 0.3640 | **0.2067** |
| Lexical only | 0.2989 | 0.1387 |

Hybrid retrieval was selected for production — best hit rate, effectively tied MRR with vector-only, and more robust across query types.

## LLM Evaluation

LLM-as-judge scoring on relevance and faithfulness (1–5 scale) across candidate prompts, 30 samples — see [eval/evaluate_llm.py](eval/evaluate_llm.py).

| Prompt | Relevance | Faithfulness | Combined |
|---|---|---|---|
| **Prompt A (selected)** | 4.4 | **4.933** | **4.667** |
| Prompt B | 4.4 | 4.867 | 4.633 |

Prompt A was selected — same relevance, higher faithfulness, and fewer tokens per call.

## Interface

- **FastAPI** backend exposing the RAG pipeline as an API ([api/main.py](api/main.py))
- **Streamlit** UI with dropdowns for company/ticker and filing period selection ([app.py](app.py))

## Ingestion Pipeline

Automated with **Prefect** ([ingest.py](ingest.py)): each ticker/year is a `@task` (fetch from EDGAR → parse → chunk), embedding and pgvector upsert are separate tasks, all orchestrated by an `ingest-filings` `@flow` with automatic retries on the EDGAR fetch step.

```bash
python ingest.py --ticker AAPL --ticker MSFT --year 2023
```

## Monitoring

- **User feedback** — thumbs up/down on every answer, stored in `query_feedback` (Postgres)
- **Request logging** — every `/query` call logs retrieval latency, LLM latency, token usage, and filters to `query_log` (Postgres), see [api/main.py](api/main.py)
- **Grafana dashboard** (`docker-compose up grafana`, [grafana/provisioning](grafana/provisioning)) with 7 panels: total queries, satisfaction rate, avg retrieval/LLM latency, queries over time, feedback breakdown, top tickers queried, and token usage over time

## Best Practices Implemented

- ✅ Hybrid search (lexical + semantic, evaluated head-to-head — see [Retrieval Evaluation](#retrieval-evaluation))
- ✅ Document re-ranking (LLM re-ranks the RRF candidate pool — [search/rerank.py](search/rerank.py))
- ✅ Query rewriting (LLM rewrites the question before retrieval — [agent/rag.py](agent/rag.py))

## Getting Started

### Prerequisites
- Docker and Docker Compose (recommended), **or** Python 3.12 + Postgres with the `pgvector` extension
- An [Anthropic API key](https://console.anthropic.com/)

### Option A: Docker Compose (recommended)

```bash
git clone https://github.com/arnenyeck06/earnings-intelligence.git
cd earnings-intelligence
cp .env.example .env              # fill in ANTHROPIC_API_KEY
docker compose up --build
```

- API: [http://localhost:8000](http://localhost:8000) (docs at `/docs`)
- UI: [http://localhost:8501](http://localhost:8501)
- Grafana: [http://localhost:3000](http://localhost:3000) (admin/admin)

The chunked corpus (`data/chunks.json`) and the embedding model (`models/Xenova/all-MiniLM-L6-v2`) are committed to the repo, so the API can serve lexical + semantic search over the existing corpus immediately. To index it into pgvector, or to ingest additional filings, run ingestion (below) against the running `postgres` container.

### Try It Out

Once `docker compose up --build` is running, verify each piece:

**API health check** — confirms the corpus loaded (should return `{"status":"ok","chunks":1842}`):
```bash
curl http://localhost:8000/health
```

**Streamlit UI** — ask a question, filter by ticker/year/section, leave 👍/👎 feedback:
```bash
open http://localhost:8501
```

**Grafana dashboard** — query volume, latency, tokens, and feedback (login `admin` / `admin`; fills in once you've run a few queries in the UI):
```bash
open http://localhost:3000
```

**Tail logs** if something looks off:
```bash
docker compose logs -f api
```

**Tear down** when done (add `-v` to also wipe the Postgres/Grafana volumes for a clean slate next run):
```bash
docker compose down
```

### Option B: Local Python

```bash
git clone https://github.com/arnenyeck06/earnings-intelligence.git
cd earnings-intelligence
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # fill in ANTHROPIC_API_KEY, Postgres connection

docker compose up postgres -d      # or point POSTGRES_* env vars at your own instance
python ingest.py --ticker AAPL --ticker MSFT --ticker NVDA --ticker GOOGL --ticker META --year 2023

uvicorn api.main:app --reload
streamlit run app.py
```

### Evaluation

```bash
python eval/generate_ground_truth.py
python eval/evaluate_retrieval.py
python eval/evaluate_llm.py
```

## Project Status

| Phase | Status |
|---|---|
| Ingestion pipeline (Prefect) | ✅ Complete |
| Hybrid retrieval (RRF) | ✅ Complete |
| Query rewriting + re-ranking | ✅ Complete |
| Retrieval evaluation | ✅ Complete |
| LLM evaluation | ✅ Complete |
| FastAPI + Streamlit interface | ✅ Complete |
| Monitoring (feedback + Grafana) | ✅ Complete |
| Docker Compose (full stack) | ✅ Complete |
| Cloud deployment | 🔲 Planned |

## Author

Arne Nyeck Nyeck — [GitHub](https://github.com/arnenyeck06) · [LinkedIn](https://www.linkedin.com/in/arne-nyeck-nyeck-539369ba/)

## License

[MIT](LICENSE)
