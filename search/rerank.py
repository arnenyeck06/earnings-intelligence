import re

RERANK_MODEL = "claude-haiku-4-5-20251001"

RERANK_PROMPT = """Question: {query}

Below are {n} candidate excerpts from SEC filings, numbered [0]-[{last}].
Rank them from most to least relevant to answering the question.
Respond with ONLY a comma-separated list of the indices, most relevant first \
(e.g. "3,0,4,1,2"). Include every index exactly once.

{listing}"""


def _format_listing(candidates, snippet_chars=500):
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"[{i}] {c['ticker']} {c['year']} — {c['section']}\n{c['text'][:snippet_chars]}"
        )
    return "\n\n".join(lines)


def rerank(query, candidates, client, num_results=5, model=RERANK_MODEL):
    """Re-rank RRF candidates with an LLM relevance pass.

    Falls back to the original (RRF) order, truncated to num_results, on any
    error so a rerank failure never breaks retrieval.
    """
    if not candidates:
        return candidates
    if len(candidates) <= num_results:
        return candidates

    prompt = RERANK_PROMPT.format(
        query=query,
        n=len(candidates),
        last=len(candidates) - 1,
        listing=_format_listing(candidates),
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        order = [int(x) for x in re.findall(r"\d+", text)]
        order = [i for i in order if 0 <= i < len(candidates)]
        seen = set()
        order = [i for i in order if not (i in seen or seen.add(i))]
        for i in range(len(candidates)):
            if i not in order:
                order.append(i)
        return [candidates[i] for i in order[:num_results]]
    except Exception:
        return candidates[:num_results]
