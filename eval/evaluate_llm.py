"""
LLM Evaluation — compare 2 prompt strategies using Claude as judge.
Prompt A: basic answer from context
Prompt B: chain-of-thought with citation requirement
"""

import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import anthropic
from search.minsearch import Index
from search.hybrid import hybrid_search
from ingestion.embedder import Embedder

CHUNKS_PATH = "data/chunks.json"
GROUND_TRUTH_PATH = "eval/ground_truth.json"
SAMPLE_SIZE = 30
OUTPUT_PATH = "eval/llm_eval_results.json"

PROMPT_A = """You are a financial analyst assistant.
Answer the question based ONLY on the provided SEC filing excerpts.
If the excerpts don't contain enough information, say so clearly."""

PROMPT_B = """You are a financial analyst assistant specializing in SEC filings.
Think step by step:
1. Identify which excerpts are most relevant to the question
2. Extract the key facts from those excerpts
3. Formulate a precise answer with citations

Answer based ONLY on the provided excerpts.
Always cite: company name, year, and section for each fact you use.
If the excerpts don't contain enough information, say so explicitly."""

JUDGE_PROMPT = """You are evaluating a financial RAG system answer.

Question: {question}

Context provided to the system:
{context}

System answer: {answer}

Rate this answer on two dimensions (1-5 scale):
1. Relevance: Does the answer address the question?
2. Faithfulness: Is the answer grounded in the provided context?

Return ONLY a JSON object like this:
{{"relevance": 4, "faithfulness": 5, "reasoning": "one sentence"}}"""


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c['ticker']} {c['year']} — {c['section']}\n{c['text'][:400]}")
    return "\n\n---\n\n".join(parts)


def get_answer(question, chunks, system_prompt, client):
    context = build_context(chunks)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return response.content[0].text, context


def judge_answer(question, context, answer, client):
    prompt = JUDGE_PROMPT.format(question=question, context=context[:2000], answer=answer)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return json.loads(response.content[0].text)
    except Exception:
        return {"relevance": 0, "faithfulness": 0, "reasoning": "parse error"}


def main():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    sample = random.sample(ground_truth, min(SAMPLE_SIZE, len(ground_truth)))

    index = Index(text_fields=["text"], keyword_fields=["ticker", "year", "section"])
    index.fit(chunks)
    embedder = Embedder()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    results = {"prompt_a": [], "prompt_b": []}

    for i, record in enumerate(sample):
        print(f"[{i+1}/{len(sample)}] {record['ticker']} {record['year']} — {record['section']}")

        retrieved = hybrid_search(
            query=record["question"], index=index,
            embedder=embedder, num_results=5
        )

        # Prompt A
        answer_a, context = get_answer(record["question"], retrieved, PROMPT_A, client)
        score_a = judge_answer(record["question"], context, answer_a, client)
        results["prompt_a"].append(score_a)

        # Prompt B
        answer_b, context = get_answer(record["question"], retrieved, PROMPT_B, client)
        score_b = judge_answer(record["question"], context, answer_b, client)
        results["prompt_b"].append(score_b)

    # Summarize
    def avg(scores, key):
        return round(sum(s[key] for s in scores) / len(scores), 3)

    print("\n" + "=" * 50)
    print("LLM EVALUATION RESULTS")
    print("=" * 50)
    for prompt, scores in results.items():
        r = avg(scores, "relevance")
        f = avg(scores, "faithfulness")
        print(f"{prompt} | Relevance: {r} | Faithfulness: {f} | Combined: {round((r+f)/2, 3)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
