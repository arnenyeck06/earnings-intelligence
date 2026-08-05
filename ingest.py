import argparse
import json
import os

from prefect import flow, task

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


@task(name="fetch-and-parse-filing", retries=2, retry_delay_seconds=5)
def fetch_and_parse(ticker, year):
    filing = fetch_10k(ticker, year)
    return parse_filing(filing)


@task(name="embed-chunks")
def embed_chunks(chunks):
    from ingestion.embedder import Embedder

    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    return embedder.encode_batch(texts).tolist()


@task(name="upsert-chunks")
def upsert_chunks_task(chunks, embeddings):
    from search.vector_store import upsert_chunks

    upsert_chunks(chunks, embeddings)


@flow(name="ingest-filings")
def ingest_flow(tickers, year, output=CHUNKS_PATH, skip_vectors=False):
    existing = load_existing(output)
    print(f"[ingest] {len(existing)} existing chunks.")

    new_chunks = []
    for ticker in tickers:
        ticker = ticker.upper()
        if already_indexed(existing, ticker, year):
            print(f"[ingest] {ticker} {year} already indexed.")
            continue
        try:
            chunks = fetch_and_parse(ticker, year)
            new_chunks.extend(chunks)
            print(f"[ingest] {ticker} {year}: {len(chunks)} chunks.")
        except Exception as e:
            print(f"[ingest] ERROR {ticker} {year}: {e}")

    if not new_chunks:
        print("[ingest] No new chunks.")
        return

    all_chunks = existing + new_chunks
    save_chunks(all_chunks, output)
    print(f"[ingest] {len(all_chunks)} total chunks saved.")

    if not skip_vectors:
        print(f"[ingest] Embedding {len(new_chunks)} chunks...")
        embeddings = embed_chunks(new_chunks)
        upsert_chunks_task(new_chunks, embeddings)
        print(f"[ingest] Upserted to pgvector.")
    else:
        print("[ingest] Skipping vector upsert.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", action="append", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", default=CHUNKS_PATH)
    parser.add_argument("--skip-vectors", action="store_true")
    args = parser.parse_args()

    ingest_flow(args.ticker, args.year, args.output, args.skip_vectors)


if __name__ == "__main__":
    main()
