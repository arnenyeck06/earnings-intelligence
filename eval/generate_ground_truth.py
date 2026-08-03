"""
Generate ground truth Q&A pairs for retrieval evaluation.
For each chunk, ask Claude to generate 3 questions answered by that chunk.
Label each question with the chunk id.
"""

import json
import os
import random
import anthropic

CHUNKS_PATH = "data/chunks.json"
OUTPUT_PATH = "eval/ground_truth.json"
QUESTIONS_PER_CHUNK = 3
SAMPLE_CHUNKS = 100  # sample to keep cost low


def generate_questions(chunk: dict, client: anthropic.Anthropic) -> list[str]:
    prompt = f"""You are evaluating a financial document retrieval system.

Given this excerpt from a SEC 10-K filing, generate {QUESTIONS_PER_CHUNK} questions that:
- Are answered directly by this excerpt
- Use different wording than the excerpt
- Range from specific to broad
- Sound like real analyst or investor questions

Company: {chunk['ticker']}
Year: {chunk['year']}
Section: {chunk['section']}

Excerpt:
{chunk['text'][:800]}

Return ONLY a JSON array of {QUESTIONS_PER_CHUNK} questions, no other text.
Example: ["question 1", "question 2", "question 3"]"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        questions = json.loads(response.content[0].text)
        return questions if isinstance(questions, list) else []
    except Exception:
        return []


def main():
    os.makedirs("eval", exist_ok=True)

    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    # Sample chunks evenly across tickers and sections
    from collections import defaultdict
    by_ticker_section = defaultdict(list)
    for chunk in chunks:
        key = (chunk["ticker"], chunk["section"])
        by_ticker_section[key].append(chunk)

    sampled = []
    per_group = max(1, SAMPLE_CHUNKS // len(by_ticker_section))
    for group_chunks in by_ticker_section.values():
        sampled.extend(random.sample(group_chunks, min(per_group, len(group_chunks))))

    sampled = sampled[:SAMPLE_CHUNKS]
    print(f"Sampled {len(sampled)} chunks for ground truth generation.")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    ground_truth = []
    for i, chunk in enumerate(sampled):
        print(f"[{i+1}/{len(sampled)}] {chunk['ticker']} {chunk['year']} — {chunk['section']}")
        questions = generate_questions(chunk, client)
        for q in questions:
            ground_truth.append({
                "question": q,
                "chunk_id": chunk["id"],
                "ticker": chunk["ticker"],
                "year": chunk["year"],
                "section": chunk["section"],
            })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nGenerated {len(ground_truth)} ground truth pairs → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
