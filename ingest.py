import argparse
import json
import os
import sys

from ingestion.edgar_client import fetch_10k
from ingestion.parser import parse_filing

CHUNKS_PATH = "data/chunks.json"


def load_existing(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_chunks(chunks, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(chunks, f, indent=2)


def already_indexed(existing, ticker, year):
    return any(c["ticker"] == ticker.upper() and c["year"] == year for c in existing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", action="append", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", default=CHUNKS_PATH)
    parser.add_argument("--skip-vectors", action="store_true")
    args = parser.parse_args()

    existing = load_existing(args.output)
    print(f"[ingest] {len(existing)} existing chunks.")

    try:
        from ingestion.embedder import Embedder
        embedder = Embedder()
        print("[ingest] Embedder ready.")
    except Exception as e:
        print(f"[ingest] Embedder not available: {e}")
        embedder = None

    new_chunks = []
    for ticker in args.ticker:
        ticker = ticker.upper()
        if already_indexed(existing, ticker, args.year):
            print(f"[ingest] {ticker} {args.year} already indexed.")
            continue
        try:
            filing = fetch_10k(ticker, args.year)
            chunks = parse_filing(filing)
            new_chunks.extend(chunks)
            print(f"[ingest] {ticker} {args.year}: {len(chunks)} chunks.")
        except Exception as e:
            print(f"[ingest] ERROR {ticker} {args.year}: {e}")

    if not new_chunks:
        print("[ingest] No new chunks.")
        return

    all_chunks = existing + new_chunks
    save_chunks(all_chunks, args.output)
    print(f"[ingest] {len(all_chunks)} total chunks saved.")

    if embedder and not args.skip_vectors:
        from search.vector_store import upsert_chunks
        print(f"[ingest] Embedding {len(new_chunks)} chunks...")
        texts = [c["text"] for c in new_chunks]
        embeddings = embedder.encode_batch(texts).tolist()
        upsert_chunks(new_chunks, embeddings)
        print(f"[ingest] Upserted to pgvector.")
    else:
        print("[ingest] Skipping vector upsert.")


if __name__ == "__main__":
    main()
