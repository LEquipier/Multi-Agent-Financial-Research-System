from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.core.config import settings


class VectorStore:
    def __init__(self) -> None:
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None
        self._embedding_fn: Any = None

    def _get_embedding_fn(self) -> Any:
        if self._embedding_fn is None:
            from langchain_openai import OpenAIEmbeddings

            self._embedding_fn = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key,
            )
        return self._embedding_fn

    async def add_documents(
        self, texts: list[str], metadatas: list[dict[str, Any]] | None = None
    ) -> None:
        if not texts:
            return

        fn = self._get_embedding_fn()
        new_embeddings = await fn.aembed_documents(texts)
        new_arr = np.array(new_embeddings, dtype=np.float32)

        if self._embeddings is None:
            self._embeddings = new_arr
        else:
            self._embeddings = np.vstack([self._embeddings, new_arr])

        self._texts.extend(texts)
        self._metadatas.extend(metadatas or [{} for _ in texts])

    async def search(
        self, query: str, k: int = 5
    ) -> list[tuple[str, dict[str, Any], float]]:
        if self._embeddings is None or len(self._texts) == 0:
            return []

        fn = self._get_embedding_fn()
        query_emb = await fn.aembed_query(query)
        query_arr = np.array(query_emb, dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_arr)
        if query_norm == 0:
            return []

        similarities = self._embeddings @ query_arr / (norms * query_norm + 1e-10)
        top_k = min(k, len(self._texts))
        indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            (self._texts[i], self._metadatas[i], float(similarities[i]))
            for i in indices
        ]

    def clear(self) -> None:
        self._texts.clear()
        self._metadatas.clear()
        self._embeddings = None


class BM25Index:
    def __init__(self) -> None:
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._tokenized: list[list[str]] = []
        self._index: BM25Okapi | None = None

    def add_documents(
        self, texts: list[str], metadatas: list[dict[str, Any]] | None = None
    ) -> None:
        if not texts:
            return

        for text in texts:
            tokens = text.lower().split()
            self._tokenized.append(tokens)

        self._texts.extend(texts)
        self._metadatas.extend(metadatas or [{} for _ in texts])
        self._index = BM25Okapi(self._tokenized)

    def search(
        self, query: str, k: int = 5
    ) -> list[tuple[str, dict[str, Any], float]]:
        if self._index is None or not self._texts:
            return []

        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_k = min(k, len(self._texts))
        indices = np.argsort(scores)[-top_k:][::-1]

        return [
            (self._texts[i], self._metadatas[i], float(scores[i]))
            for i in indices
            if scores[i] > 0
        ]

    def clear(self) -> None:
        self._texts.clear()
        self._metadatas.clear()
        self._tokenized.clear()
        self._index = None


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        time_decay_lambda: float = 0.1,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.time_decay_lambda = time_decay_lambda

    def _time_decay(self, published_at: Any) -> float:
        if not published_at:
            return 1.0

        if isinstance(published_at, (int, float)):
            pub_dt = datetime.fromtimestamp(published_at, tz=timezone.utc)
        elif isinstance(published_at, str):
            pub_dt = datetime.fromisoformat(published_at).replace(tzinfo=timezone.utc)
        elif isinstance(published_at, datetime):
            pub_dt = published_at if published_at.tzinfo else published_at.replace(tzinfo=timezone.utc)
        else:
            return 1.0

        now = datetime.now(tz=timezone.utc)
        days_old = max(0, (now - pub_dt).total_seconds() / 86400)
        return math.exp(-self.time_decay_lambda * days_old)

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> list[dict[str, Any]]:
        vector_results = await self.vector_store.search(query, k=k * 2)
        bm25_results = self.bm25_index.search(query, k=k * 2)

        # Reciprocal Rank Fusion
        scores: dict[int, float] = {}
        text_map: dict[int, tuple[str, dict]] = {}
        rrf_k = 60

        for rank, (text, meta, _score) in enumerate(vector_results):
            idx = hash(text)
            scores[idx] = scores.get(idx, 0) + vector_weight / (rrf_k + rank + 1)
            text_map[idx] = (text, meta)

        for rank, (text, meta, _score) in enumerate(bm25_results):
            idx = hash(text)
            scores[idx] = scores.get(idx, 0) + bm25_weight / (rrf_k + rank + 1)
            text_map[idx] = (text, meta)

        # Apply time decay
        for idx in scores:
            text, meta = text_map[idx]
            decay = self._time_decay(meta.get("published_at"))
            scores[idx] *= decay

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]

        return [
            {
                "text": text_map[idx][0],
                "metadata": text_map[idx][1],
                "score": round(scores[idx], 6),
            }
            for idx in sorted_ids
        ]


class RAGPipeline:
    def __init__(self, time_decay_lambda: float = 0.1) -> None:
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        self.hybrid_retriever = HybridRetriever(
            self.vector_store, self.bm25_index, time_decay_lambda
        )

    async def ingest_news(self, articles: list[dict[str, Any]]) -> int:
        texts = []
        metadatas = []
        for article in articles:
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            text = f"{headline}. {summary}".strip()
            if not text or text == ".":
                continue
            texts.append(text)
            metadatas.append({
                "source": article.get("source", ""),
                "published_at": article.get("published_at", ""),
                "url": article.get("url", ""),
            })

        if texts:
            await self.vector_store.add_documents(texts, metadatas)
            self.bm25_index.add_documents(texts, metadatas)

        return len(texts)

    async def ingest_market_summary(self, symbol: str, summary: str) -> None:
        metadata = {
            "symbol": symbol,
            "published_at": datetime.now(tz=timezone.utc).isoformat(),
            "type": "market_summary",
        }
        await self.vector_store.add_documents([summary], [metadata])
        self.bm25_index.add_documents([summary], [metadata])

    async def retrieve_context(
        self, query: str, k: int = 5, recency_bias: float = 0.1
    ) -> str:
        self.hybrid_retriever.time_decay_lambda = recency_bias
        results = await self.hybrid_retriever.retrieve(query, k=k)

        if not results:
            return "No relevant context found."

        context_parts = []
        for r in results:
            context_parts.append(
                f"[Score: {r['score']:.4f}] {r['text']}"
            )

        return "\n\n".join(context_parts)

    def clear(self) -> None:
        self.vector_store.clear()
        self.bm25_index.clear()
