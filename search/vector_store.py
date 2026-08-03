import os
import psycopg2
import psycopg2.extras
from typing import Optional


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "financial_rag"),
        user=os.getenv("POSTGRES_USER", "rag"),
        password=os.getenv("POSTGRES_PASSWORD", "rag123"),
    )


def upsert_chunks(chunks, embeddings):
    conn = get_conn()
    cur = conn.cursor()
    for chunk, embedding in zip(chunks, embeddings):
        cur.execute(
            """
            INSERT INTO chunks (id, ticker, year, filing_date, doc_type, section, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding
            """,
            (
                chunk["id"], chunk["ticker"], chunk["year"],
                chunk.get("filing_date"), chunk.get("doc_type", "10-K"),
                chunk.get("section"), chunk.get("chunk_index", 0),
                chunk["text"], str(embedding),
            ),
        )
    conn.commit()
    cur.close()
    conn.close()


def vector_search(query_embedding, ticker=None, year=None, section=None, num_results=10):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    filters = []
    params = [str(query_embedding)]
    if ticker:
        filters.append("ticker = %s")
        params.append(ticker)
    if year:
        filters.append("year = %s")
        params.append(year)
    if section:
        filters.append("section = %s")
        params.append(section)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.extend([str(query_embedding), num_results])
    cur.execute(
        f"""
        SELECT id, ticker, year, filing_date, doc_type, section, chunk_index, text,
               1 - (embedding <=> %s::vector) AS _score
        FROM chunks
        {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        params,
    )
    results = [dict(row) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results
