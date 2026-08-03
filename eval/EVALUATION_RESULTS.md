# Retrieval & LLM Evaluation Results

## Retrieval Evaluation

Ground truth: 261 Q&A pairs generated across AAPL, MSFT, NVDA, GOOGL, META (2019-2023)

| Method  | Hit Rate | MRR    |
|---------|----------|--------|
| Keyword | 0.2989   | 0.1387 |
| Vector  | 0.3640   | 0.2067 |
| Hybrid  | 0.3678   | 0.2065 |

**Winner: Hybrid RRF** — best hit rate, tied MRR with vector, more robust.

## LLM Evaluation

30 samples evaluated with Claude-as-judge (1-5 scale).

| Prompt   | Relevance | Faithfulness | Combined |
|----------|-----------|--------------|----------|
| Prompt A | 4.4       | 4.933        | 4.667    |
| Prompt B | 4.4       | 4.867        | 4.633    |

**Winner: Prompt A** — simpler, higher faithfulness, fewer tokens per call.

## Decisions

- Default search: **hybrid RRF** (minsearch + pgvector + RRF k=60)
- Default prompt: **Prompt A** (basic answer from context with citation requirement)
