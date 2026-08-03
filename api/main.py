"""
FastAPI interface for Earnings Intelligence Platform.

Endpoints:
  POST /query     — ask a question, get a grounded answer
  GET  /filings   — list indexed companies and years
  POST /feedback  — submit thumbs up/down on an answer
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import anthropic

from search.minsearch import Index
from search.hybrid import hybrid_search
from ingestion.embedder import Embedder
from search.vector_store import get_conn

CHUNKS_PATH = "data/chunks.json"

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in SEC filings.
Answer questions based ONLY on the provided SEC filing excerpts.
Always cite which company, year, and section your answer comes from.
If the excerpts don't contain enough information to answer, say so clearly."""

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


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    feedback: int  # 1 = thumbs up, -1 = thumbs down
    ticker: Optional[str] = None
    year: Optional[int] = None


# --- Endpoints ---

@app.post("/query")
def query(req: QueryRequest):
    retrieved = hybrid_search(
        query=req.question,
        index=index,
        embedder=embedder,
        ticker=req.ticker.upper() if req.ticker else None,
        year=req.year,
        section=req.section,
        num_results=req.num_results,
    )

    context_parts = []
    for i, c in enumerate(retrieved, 1):
        context_parts.append(
            f"[{i}] {c['ticker']} {c['year']} 10-K — {c['section']}\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"SEC Filing Excerpts:\n{context}\n\nQuestion: {req.question}"
        }],
    )

    return {
        "answer": response.content[0].text,
        "sources": [
            {
                "ticker": c["ticker"],
                "year": c["year"],
                "section": c["section"],
                "chunk_index": c["chunk_index"],
            }
            for c in retrieved
        ],
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


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
