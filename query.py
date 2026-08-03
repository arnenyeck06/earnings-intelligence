import argparse
import json
import os
import sys

from search.minsearch import Index
from agent.rag import answer_query

CHUNKS_PATH = "data/chunks.json"


def load_chunks(path):
    if not os.path.exists(path):
        print(f"[query] No chunks at {path}. Run ingest.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_index(chunks):
    idx = Index(text_fields=["text"], keyword_fields=["ticker", "year", "section"])
    idx.fit(chunks)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Natural language question")
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--year", type=int)
    parser.add_argument("--section")
    parser.add_argument("--search", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--num-results", type=int, default=5)
    parser.add_argument("--chunks", default=CHUNKS_PATH)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    index = build_index(chunks)

    ticker = args.ticker[0].upper() if args.ticker and len(args.ticker) == 1 else None
    filter_dict = {}
    if ticker:
        filter_dict["ticker"] = ticker
    if args.year:
        filter_dict["year"] = args.year

    if args.search == "keyword":
        retrieved = index.search(args.query, filter_dict=filter_dict, num_results=args.num_results)

    elif args.search == "vector":
        from ingestion.embedder import Embedder
        from search.vector_store import vector_search
        embedder = Embedder()
        q_vec = embedder.encode(args.query)
        retrieved = vector_search(q_vec, ticker=ticker, year=args.year, num_results=args.num_results)

    else:  # hybrid
        try:
            from ingestion.embedder import Embedder
            from search.hybrid import hybrid_search
            embedder = Embedder()
            retrieved = hybrid_search(
                query=args.query, index=index, embedder=embedder,
                ticker=ticker, year=args.year, section=args.section,
                num_results=args.num_results,
            )
        except Exception as e:
            print(f"[query] Hybrid failed ({e}), falling back to keyword.")
            retrieved = index.search(args.query, filter_dict=filter_dict, num_results=args.num_results)

    result = answer_query(args.query, retrieved)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])
    print("\n" + "-" * 60)
    print(f"SOURCES ({args.search})")
    print("-" * 60)
    for s in result["sources"]:
        print(f"  {s['ticker']} {s['year']} — {s['section']} (chunk {s['chunk_index']})")
    print(f"\n[tokens] in={result.get('input_tokens','?')} out={result.get('output_tokens','?')}")


if __name__ == "__main__":
    main()
