"""Recuperação: BM25, vetores de n-gramas de caracteres, fusão RRF e reranker lexical.

`CharNgramEmbedder` e `LexicalReranker` são substitutos offline: trocar por bge-m3 /
bge-reranker exige só implementar `encode()` e `score()` (ver docs/adr/0001).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import Chunk
from .text import content_stems

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    confidence: float  # 0..1, usado para decidir recusa


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [{} for _ in docs]
        df: dict[str, int] = {}
        for i, d in enumerate(docs):
            for t in d:
                self.tf[i][t] = self.tf[i].get(t, 0) + 1
            for t in self.tf[i]:
                df[t] = df.get(t, 0) + 1
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean() if self.n else 1.0
        self.idf = {t: np.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q: list[str]) -> np.ndarray:
        out = np.zeros(self.n)
        for i in range(self.n):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            for t in q:
                f = self.tf[i].get(t, 0)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return out


class CharNgramEmbedder:
    def __init__(self):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True, strip_accents="unicode")

    def fit(self, texts: list[str]) -> "CharNgramEmbedder":
        self.matrix = self.vec.fit_transform(texts)
        return self

    def similarities(self, query: str) -> np.ndarray:
        return (self.matrix @ self.vec.transform([query]).T).toarray().ravel()


def _coverage(q: list[str], d: set[str]) -> float:
    return sum(t in d for t in q) / len(q) if q else 0.0


def _bigram_coverage(q: list[str], d: list[str]) -> float:
    qb = set(zip(q, q[1:]))
    return len(qb & set(zip(d, d[1:]))) / len(qb) if qb else 0.0


class Retriever:
    def __init__(self, chunks: list[Chunk], mode: str = "hybrid_rerank", rrf_k: int = 60, rerank_pool: int = 6):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.chunks, self.mode, self.rrf_k, self.pool = chunks, mode, rrf_k, rerank_pool
        self.stems = [content_stems(c.text) for c in chunks]
        self.bm25 = BM25(self.stems)
        self.dense = CharNgramEmbedder().fit([c.text for c in chunks])

    def _rank(self, scores: np.ndarray) -> np.ndarray:
        return np.argsort(-scores, kind="stable")

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = content_stems(query)
        b, d = self.bm25.scores(q), self.dense.similarities(query)
        if self.mode == "bm25":
            final = b
        elif self.mode == "dense":
            final = d
        else:
            final = np.zeros(len(self.chunks))
            for s in (b, d):
                for r, i in enumerate(self._rank(s)):
                    final[i] += 1 / (self.rrf_k + r + 1)
        order = list(self._rank(final))
        if self.mode == "hybrid_rerank":
            pool = order[: self.pool]
            rr = {i: 0.7 * _coverage(q, set(self.stems[i])) + 0.3 * _bigram_coverage(q, self.stems[i]) + 0.5 * d[i] for i in pool}
            order = sorted(pool, key=lambda i: -rr[i]) + order[self.pool :]
            final = final.copy()
            for i in pool:
                final[i] = rr[i]
        hits = []
        for i in order[:k]:
            conf = 0.6 * _coverage(q, set(self.stems[i])) + 0.4 * min(1.0, d[i] / 0.35)
            hits.append(Hit(self.chunks[i], float(final[i]), float(min(1.0, conf))))
        return hits
