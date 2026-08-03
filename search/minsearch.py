import math
from collections import defaultdict


class Index:
    def __init__(self, text_fields, keyword_fields=None):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields or []
        self.docs = []
        self.index = defaultdict(lambda: defaultdict(list))
        self.doc_count = 0

    def fit(self, docs):
        self.docs = docs
        self.doc_count = len(docs)
        self.index = defaultdict(lambda: defaultdict(list))
        for doc_id, doc in enumerate(docs):
            for field in self.text_fields:
                text = doc.get(field, "")
                for term in set(self._tokenize(text)):
                    self.index[term][field].append(doc_id)
        return self

    def search(self, query, filter_dict=None, boost_dict=None, num_results=10):
        filter_dict = filter_dict or {}
        boost_dict = boost_dict or {f: 1.0 for f in self.text_fields}
        scores = defaultdict(float)
        for term in self._tokenize(query):
            for field in self.text_fields:
                if term not in self.index or field not in self.index[term]:
                    continue
                matching = self.index[term][field]
                idf = math.log((self.doc_count + 1) / (len(matching) + 1)) + 1
                boost = boost_dict.get(field, 1.0)
                for doc_id in matching:
                    scores[doc_id] += idf * boost
        results = []
        for doc_id, score in sorted(scores.items(), key=lambda x: -x[1]):
            doc = self.docs[doc_id]
            if self._matches_filter(doc, filter_dict):
                results.append({**doc, "_score": round(score, 4)})
            if len(results) >= num_results:
                break
        return results

    def _matches_filter(self, doc, filter_dict):
        for key, value in filter_dict.items():
            doc_val = doc.get(key)
            if isinstance(value, list):
                if doc_val not in value:
                    return False
            else:
                if str(doc_val) != str(value):
                    return False
        return True

    def _tokenize(self, text):
        import re
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if len(t) > 2]
