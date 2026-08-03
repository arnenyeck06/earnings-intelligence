from search.minsearch import Index
from search.vector_store import vector_search as pgvector_search


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            docs[doc_id] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [
        {**docs[doc_id], "_rrf_score": round(scores[doc_id], 6)}
        for doc_id in ranked[:num_results]
    ]


def hybrid_search(query, index, embedder, ticker=None, year=None, section=None, num_results=5, k=60):
    filter_dict = {}
    if ticker:
        filter_dict["ticker"] = ticker
    if year:
        filter_dict["year"] = year
    if section:
        filter_dict["section"] = section

    keyword_results = index.search(
        query=query, filter_dict=filter_dict,
        boost_dict={"text": 1.0}, num_results=num_results * 2,
    )

    query_vec = embedder.encode(query)
    vector_results = pgvector_search(
        query_embedding=query_vec, ticker=ticker,
        year=year, section=section, num_results=num_results * 2,
    )

    return rrf([keyword_results, vector_results], k=k, num_results=num_results)
