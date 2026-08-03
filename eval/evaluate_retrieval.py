"""
Evaluate keyword, vector, and hybrid search against ground truth.
Metrics: Hit Rate and MRR.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tqdm import tqdm
from search.minsearch import Index
from search.vector_store import vector_search
from search.hybrid import hybrid_search
from ingestion.embedder import Embedder

CHUNKS_PATH = "data/chunks.json"
GROUND_TRUTH_PATH = "eval/ground_truth.json"


def load_data():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)
    return chunks, ground_truth


def build_index(chunks):
    idx = Index(text_fields=["text"], keyword_fields=["ticker", "year", "section"])
    idx.fit(chunks)
    return idx


def hit_rate(relevance_total):
    return sum(any(r) for r in relevance_total) / len(relevance_total)


def mrr(relevance_total):
    total = 0.0
    for relevance in relevance_total:
        for rank, r in enumerate(relevance):
            if r == 1:
                total += 1 / (rank + 1)
                break
    return total / len(relevance_total)


def compute_relevance(results, chunk_id, num_results=5):
    result_ids = [r["id"] for r in results[:num_results]]
    return [1 if rid == chunk_id else 0 for rid in result_ids]


def evaluate(ground_truth, search_fn, num_results=5):
    relevance_total = []
    for record in tqdm(ground_truth):
        results = search_fn(record["question"])
        relevance = compute_relevance(results, record["chunk_id"], num_results)
        relevance_total.append(relevance)
    return {
        "hit_rate": round(hit_rate(relevance_total), 4),
        "mrr": round(mrr(relevance_total), 4),
    }


def main():
    print("Loading data...")
    chunks, ground_truth = load_data()
    print(f"{len(chunks)} chunks, {len(ground_truth)} ground truth pairs.")

    print("Building keyword index...")
    index = build_index(chunks)

    print("Loading embedder...")
    embedder = Embedder()

    # Define search functions
    def keyword_search(query):
        return index.search(query, num_results=5)

    def vec_search(query):
        q_vec = embedder.encode(query)
        return vector_search(q_vec, num_results=5)

    def hybrid(query):
        return hybrid_search(
            query=query, index=index, embedder=embedder, num_results=5
        )

    results = {}

    print("\nEvaluating keyword search...")
    results["keyword"] = evaluate(ground_truth, keyword_search)
    print(f"  Hit Rate: {results['keyword']['hit_rate']} | MRR: {results['keyword']['mrr']}")

    print("\nEvaluating vector search...")
    results["vector"] = evaluate(ground_truth, vec_search)
    print(f"  Hit Rate: {results['vector']['hit_rate']} | MRR: {results['vector']['mrr']}")

    print("\nEvaluating hybrid search...")
    results["hybrid"] = evaluate(ground_truth, hybrid)
    print(f"  Hit Rate: {results['hybrid']['hit_rate']} | MRR: {results['hybrid']['mrr']}")

    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)
    for method, scores in results.items():
        print(f"{method:10} | Hit Rate: {scores['hit_rate']:.4f} | MRR: {scores['mrr']:.4f}")

    best = max(results, key=lambda x: results[x]["mrr"])
    print(f"\nBest method: {best} (MRR={results[best]['mrr']})")

    # Save results
    with open("eval/retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to eval/retrieval_results.json")


if __name__ == "__main__":
    main()
