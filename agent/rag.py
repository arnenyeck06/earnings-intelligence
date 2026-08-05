import os
import anthropic

ANTHROPIC_MODEL = "claude-sonnet-4-6"
REWRITE_MODEL = "claude-haiku-4-5-20251001"
MAX_CONTEXT_CHUNKS = 5

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in SEC filings.
Answer questions based ONLY on the provided SEC filing excerpts.
Always cite which company, year, and section your answer comes from.
If the excerpts don't contain enough information to answer, say so clearly.
Do not use any knowledge outside of the provided excerpts."""

REWRITE_SYSTEM_PROMPT = """You rewrite user questions into search queries for a hybrid \
(keyword + semantic) search engine over SEC filings (10-Ks, 10-Qs, earnings transcripts).
Expand abbreviations, spell out company/financial terminology, and drop conversational \
filler, but keep it a single short query. Respond with ONLY the rewritten query, no \
quotes, no preamble."""


def rewrite_query(query, client=None, model=REWRITE_MODEL):
    """Rewrite a natural-language question into a better retrieval query.

    Falls back to the original query on any error so retrieval never breaks
    because of the rewrite step.
    """
    if client is None:
        client = _default_client()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=100,
            system=REWRITE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        rewritten = response.content[0].text.strip().strip('"')
        return rewritten if rewritten else query
    except Exception:
        return query


def _default_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")
    return anthropic.Anthropic(api_key=api_key)


def build_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {chunk['ticker']} {chunk['year']} 10-K — {chunk['section']}\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(query, chunks, client):
    if not chunks:
        return {"answer": "No relevant filing excerpts found.", "sources": []}
    context = build_context(chunks)
    user_message = f"""SEC Filing Excerpts:
{context}

Question: {query}

Answer based only on the excerpts above. Cite the company, year, and section."""
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return {
        "answer": response.content[0].text,
        "sources": [{"ticker": c["ticker"], "year": c["year"],
                     "section": c["section"], "chunk_index": c["chunk_index"],
                     "score": c.get("_score", 0)} for c in chunks],
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def answer_query(query, chunks, client=None):
    if client is None:
        client = _default_client()
    return generate_answer(query, chunks, client)
