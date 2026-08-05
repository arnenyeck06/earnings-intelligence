CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    ticker      TEXT NOT NULL,
    year        INTEGER NOT NULL,
    filing_date TEXT,
    doc_type    TEXT NOT NULL DEFAULT '10-K',
    section     TEXT,
    chunk_index INTEGER,
    text        TEXT NOT NULL,
    embedding   vector(384)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_ticker_idx ON chunks (ticker);
CREATE INDEX IF NOT EXISTS chunks_year_idx ON chunks (year);
CREATE INDEX IF NOT EXISTS chunks_section_idx ON chunks (section);

CREATE TABLE IF NOT EXISTS query_feedback (
    id          SERIAL PRIMARY KEY,
    query       TEXT NOT NULL,
    answer      TEXT,
    ticker      TEXT,
    year        INTEGER,
    feedback    INTEGER,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS query_log (
    id               SERIAL PRIMARY KEY,
    query            TEXT NOT NULL,
    rewritten_query  TEXT,
    ticker           TEXT,
    year             INTEGER,
    section          TEXT,
    num_results      INTEGER,
    retrieval_ms     INTEGER,
    llm_ms           INTEGER,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS query_log_created_at_idx ON query_log (created_at);
CREATE INDEX IF NOT EXISTS query_feedback_created_at_idx ON query_feedback (created_at);
