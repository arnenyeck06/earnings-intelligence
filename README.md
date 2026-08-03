# Earnings Intelligence Platform

A hybrid RAG assistant for querying SEC EDGAR filings — 10-Ks, 10-Qs, and earnings call transcripts — in plain English. Built as the capstone project for [DataTalks.Club LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp).

## Problem

Public company filings contain the information investors and analysts need — margins, risk factors, forward guidance, segment performance — but it's buried across hundreds of pages per filing, repeated quarter over quarter, in dense financial language. Finding a specific fact (e.g. "what did NVIDIA say about data center revenue growth in Q3 2023?") means manually searching PDFs or EDGAR's own interface.

This platform lets a user ask a natural-language question and get a grounded answer sourced directly from the underlying filings, using retrieval that combines keyword and semantic search rather than plain "control-F."

**Corpus:** 1,842 chunks from 10-K/10-Q filings and earnings transcripts for AAPL, MSFT, NVDA, GOOGL, and META, spanning 2019–2023, sourced from SEC EDGAR.

## Architecture

```
EDGAR API → Ingestion (parser + chunker) → Embedder → pgvector (Postgres)
                                                      ↘
                                          MinSearch (lexical) ─┐
                                          pgvector (semantic) ─┼→ RRF fusion → LLM → Answer
                                                              ┘
                                     FastAPI ⇄ Streamlit UI
```

## Retrieval Flow

1. **Lexical retrieval** — [MinSearch](https://github.com/alexeygrigorev/minsearch) over filing text
2. **Semantic retrieval** — embeddings stored in **pgvector** (Postgres)
3. **Hybrid fusion** — [Reciprocal Rank Fusion (RRF)](https://en.wikipedia.org/wiki/Reciprocal_rank_fusion) combines both ranked lists
4. Top hybrid results are passed as context to the LLM, which generates a grounded, sourced answer

## Retrieval Evaluation

| Approach | Hit Rate | MRR |
|---|---|---|
| **Hybrid (RRF)** | **0.3678** | **0.2065** |
| Lexical only | _[fill in]_ | _[fill in]_ |
| Semantic only | _[fill in]_ | _[fill in]_ |

Hybrid retrieval was selected for production based on these results.

## LLM Evaluation

LLM-as-judge scoring on relevance and faithfulness (1–5 scale) across candidate prompts:

| Prompt | Relevance | Faithfulness |
|---|---|---|
| **Prompt A (selected)** | **4.4** | **4.933** |
| Prompt B | _[fill in]_ | _[fill in]_ |

## Interface

- **FastAPI** backend exposing the RAG pipeline as an API
- **Streamlit** UI with dropdowns for company/ticker and filing period selection

![UI screenshot](docs/ui-screenshot.png)
*(drop a screenshot at `docs/ui-screenshot.png` — see checklist below)*

## Ingestion Pipeline

Semi-automated ingestion via Python scripts: EDGAR client → parser → embedder → pgvector upsert.

## Best Practices Implemented

- ✅ Hybrid search (lexical + semantic, evaluated head-to-head)
- ⬜ Document re-ranking
- ⬜ Query rewriting

## Getting Started

### Prerequisites
- Python _[your version]_
- Postgres with the `pgvector` extension
- OpenAI (or your chosen LLM provider) API key

### Setup
```bash
git clone https://github.com/arnenyeck06/earnings-intelligence.git
cd earnings-intelligence
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # or: uv sync, if using uv
cp .env.example .env              # fill in your API key and DB connection string
```

### Ingest data
```bash
python ingest.py   # adjust to your actual script name
```

### Run the API
```bash
uvicorn main:app --reload   # adjust module path if different
```

### Run the UI
```bash
streamlit run ui/app.py   # adjust path if different
```

## Project Status

| Phase | Status |
|---|---|
| Ingestion pipeline | ✅ Complete |
| Hybrid retrieval (RRF) | ✅ Complete |
| Retrieval evaluation | ✅ Complete |
| LLM evaluation | ✅ Complete |
| FastAPI + Streamlit interface | ✅ Complete |
| Docker Compose | 🔲 Planned |
| Cloud deployment | 🔲 Planned |

## Author

Arne Nyeck Nyeck — [GitHub](https://github.com/arnenyeck06) · [LinkedIn](https://linkedin.com/in/arne-nyecknyeck-539369ba)
