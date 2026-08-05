"""
FastAPI interface for Earnings Intelligence Platform.

Endpoints:
  POST /query     — ask a question, get a grounded answer
  GET  /filings   — list indexed companies and years
  POST /feedback  — submit thumbs up/down on an answer
  GET  /health    — liveness check
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import anthropic

from search.minsearch import Index
from search.hybrid import hybrid_search
from search.rerank import rerank
from search.vector_store import get_conn
from ingestion.embedder import Embedder
from agent.rag import generate_answer, rewrite_query

CHUNKS_PATH = "data/chunks.json"
RERANK_POOL_MULTIPLIER = 3  # fetch this many extra candidates for the reranker to choose from

app = FastAPI(title="Earnings Intelligence Platform", version="1.0")

# Load at startup
chunks = []
index = None
embedder = None
client = None


@app.on_event("startup")
def startup():
    global chunks, index, embedder, client

    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    index = Index(text_fields=["text"], keyword_fields=["ticker", "year", "section"])
    index.fit(chunks)

    embedder = Embedder()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print(f"[startup] {len(chunks)} chunks loaded.")


# --- Request/Response models ---

class QueryRequest(BaseModel):
    question: str
    ticker: Optional[str] = None
    year: Optional[int] = None
    section: Optional[str] = None
    num_results: int = 5
    rerank: bool = True
    rewrite: bool = True


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    feedback: int  # 1 = thumbs up, -1 = thumbs down
    ticker: Optional[str] = None
    year: Optional[int] = None


# --- Endpoints ---

@app.post("/query")
def query(req: QueryRequest):
    search_query = rewrite_query(req.question, client=client) if req.rewrite else req.question

    retrieval_start = time.perf_counter()
    pool_size = req.num_results * RERANK_POOL_MULTIPLIER if req.rerank else req.num_results
    candidates = hybrid_search(
        query=search_query,
        index=index,
        embedder=embedder,
        ticker=req.ticker.upper() if req.ticker else None,
        year=req.year,
        section=req.section,
        num_results=pool_size,
    )
    retrieved = (
        rerank(search_query, candidates, client, num_results=req.num_results)
        if req.rerank
        else candidates[: req.num_results]
    )
    retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)

    llm_start = time.perf_counter()
    result = generate_answer(req.question, retrieved, client)
    llm_ms = int((time.perf_counter() - llm_start) * 1000)

    _log_query(
        query=req.question,
        rewritten_query=search_query if req.rewrite else None,
        ticker=req.ticker,
        year=req.year,
        section=req.section,
        num_results=req.num_results,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
    )

    return result


def _log_query(**fields):
    """Best-effort request logging for the monitoring dashboard. Never fails the request."""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO query_log
                (query, rewritten_query, ticker, year, section, num_results,
                 retrieval_ms, llm_ms, input_tokens, output_tokens)
            VALUES (%(query)s, %(rewritten_query)s, %(ticker)s, %(year)s, %(section)s,
                    %(num_results)s, %(retrieval_ms)s, %(llm_ms)s, %(input_tokens)s, %(output_tokens)s)
            """,
            fields,
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[query_log] failed to log query: {e}")


@app.get("/filings")
def filings():
    from collections import defaultdict
    by_ticker = defaultdict(set)
    for c in chunks:
        by_ticker[c["ticker"]].add(c["year"])
    return {
        ticker: sorted(years, reverse=True)
        for ticker, years in sorted(by_ticker.items())
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO query_feedback (query, answer, ticker, year, feedback)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (req.query, req.answer, req.ticker, req.year, req.feedback),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(chunks)}
