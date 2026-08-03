import os
import anthropic

ANTHROPIC_MODEL = "claude-sonnet-4-6"
MAX_CONTEXT_CHUNKS = 5

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in SEC filings.
Answer questions based ONLY on the provided SEC filing excerpts.
Always cite which company, year, and section your answer comes from.
If the excerpts don't contain enough information to answer, say so clearly.
Do not use any knowledge outside of the provided excerpts."""


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
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")
        client = anthropic.Anthropic(api_key=api_key)
    return generate_answer(query, chunks, client)
