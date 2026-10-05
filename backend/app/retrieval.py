"""Hybrid retrieval over regulation passages.

Two rankers vote and their ranks are fused (reciprocal rank fusion):

  lexical  BM25 over word tokens - exact regulatory vocabulary matters
  dense    TF-IDF projected to 128 dimensions with truncated SVD (latent
           semantic analysis), searched with a FAISS inner-product index

The dense side is deliberately a local model: the index builds in under a
second, needs no download or API key, and keeps regulated text on the box.
`Embedder` is the seam for swapping in a neural sentence encoder.
"""
import re
from functools import lru_cache

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .corpus import load_corpus

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an and are as at be by for from in is it its of on or shall that the their this to "
           "with which such any each all not no may will if been has have was were than then "
           "these those there other into under per".split())


def tokenize(text):
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


def stem(t):
    """Light suffix stripping so 'records/recorded/recording' meet."""
    for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)] + ("y" if suf == "ies" else "")
    return t


class Embedder:
    """TF-IDF -> SVD -> L2-normalised vectors."""

    def __init__(self, dim=config.LSA_DIM):
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2),
                                     sublinear_tf=True, min_df=1)
        self.svd = TruncatedSVD(n_components=dim, random_state=0)

    def fit(self, texts):
        self.svd.fit(self.tfidf.fit_transform(texts))
        return self

    def encode(self, texts):
        v = self.svd.transform(self.tfidf.transform(texts)).astype("float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v


class HybridRetriever:
    def __init__(self, passages):
        self.passages = passages
        texts = [p.search_text for p in passages]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.embedder = Embedder(min(config.LSA_DIM, len(passages) - 1)).fit(texts)
        vectors = self.embedder.encode(texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query, k=config.TOP_K, mode="hybrid", pool=30):
        """Return [(passage, score)] best first. mode: hybrid | bm25 | dense."""
        pool = min(pool, len(self.passages))
        bm = self.bm25.get_scores(tokenize(query))
        bm_rank = np.argsort(-bm)[:pool]
        sims, dn_rank = self.index.search(self.embedder.encode([query]), pool)
        sims, dn_rank = sims[0], dn_rank[0]
        if mode == "bm25":
            return [(self.passages[i], float(bm[i])) for i in bm_rank[:k]]
        if mode == "dense":
            return [(self.passages[i], float(s)) for i, s in zip(dn_rank[:k], sims[:k])]
        fused = {}
        for ranks in (bm_rank, dn_rank):
            for r, i in enumerate(ranks):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (config.RRF_K + r + 1)
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(self.passages[i], float(s)) for i, s in best]


@lru_cache(maxsize=1)
def get_retriever():
    return HybridRetriever(load_corpus())
