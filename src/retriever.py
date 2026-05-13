"""
Step 2: Multi-path Retrieval
논문 3.2절: 패턴 그래프를 기반으로 텍스트/이미지/테이블 다중 경로 검색
"""

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .query_analyzer import PatternGraph


@dataclass
class RetrievedDoc:
    doc_id: str
    content: str
    modality: str            # text / image / table
    score: float
    metadata: dict = field(default_factory=dict)
    image_path: Optional[str] = None
    image_b64: Optional[str] = None  # Claude API용 base64


class MultiPathRetriever:
    """
    패턴 그래프 기반 다중 경로 검색기.
    텍스트 / 이미지 / 테이블 각 경로에서 독립적으로 검색 후 합산.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self._text_docs: list[dict] = []
        self._text_embeddings: Optional[np.ndarray] = None
        self._image_docs: list[dict] = []
        self._table_docs: list[dict] = []

    # ------------------------------------------------------------------ #
    # 문서 인덱싱
    # ------------------------------------------------------------------ #
    def index_texts(self, docs: list[dict]) -> None:
        """docs: [{"id": str, "content": str, "metadata": dict}]"""
        self._text_docs = docs
        texts = [d["content"] for d in docs]
        self._text_embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        print(f"[Retriever] {len(docs)}개 텍스트 문서 인덱싱 완료")

    def index_images(self, docs: list[dict]) -> None:
        """docs: [{"id": str, "path": str, "caption": str, "metadata": dict}]"""
        self._image_docs = docs
        print(f"[Retriever] {len(docs)}개 이미지 문서 인덱싱 완료")

    def index_tables(self, docs: list[dict]) -> None:
        """docs: [{"id": str, "content": str (markdown table), "metadata": dict}]"""
        self._table_docs = docs
        print(f"[Retriever] {len(docs)}개 테이블 문서 인덱싱 완료")

    # ------------------------------------------------------------------ #
    # 검색
    # ------------------------------------------------------------------ #
    def retrieve(self, pattern: PatternGraph, top_k: int = 5) -> list[RetrievedDoc]:
        results: list[RetrievedDoc] = []

        for subquery in pattern.subqueries:
            results.extend(self._retrieve_text(subquery, top_k))

        if "image" in pattern.required_modalities:
            results.extend(self._retrieve_images(pattern.raw_query, top_k))

        if "table" in pattern.required_modalities:
            results.extend(self._retrieve_tables(pattern.raw_query, top_k))

        # 중복 제거 후 점수 내림차순 정렬
        seen = set()
        unique: list[RetrievedDoc] = []
        for doc in sorted(results, key=lambda x: x.score, reverse=True):
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                unique.append(doc)

        return unique[:top_k * 2]

    def _retrieve_text(self, query: str, top_k: int) -> list[RetrievedDoc]:
        if self._text_embeddings is None or len(self._text_docs) == 0:
            return []

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        scores = (self._text_embeddings @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievedDoc(
                doc_id=self._text_docs[i]["id"],
                content=self._text_docs[i]["content"],
                modality="text",
                score=float(scores[i]),
                metadata=self._text_docs[i].get("metadata", {}),
            )
            for i in top_idx
        ]

    def _retrieve_images(self, query: str, top_k: int) -> list[RetrievedDoc]:
        if not self._image_docs:
            return []

        # 캡션으로 임베딩 유사도 계산
        captions = [d.get("caption", "") for d in self._image_docs]
        cap_emb = self.embedder.encode(captions, convert_to_numpy=True)
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        scores = (cap_emb @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        docs = []
        for i in top_idx:
            image_doc = self._image_docs[i]
            b64 = None
            path = image_doc.get("path")
            if path and Path(path).exists():
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            docs.append(RetrievedDoc(
                doc_id=image_doc["id"],
                content=image_doc.get("caption", ""),
                modality="image",
                score=float(scores[i]),
                metadata=image_doc.get("metadata", {}),
                image_path=path,
                image_b64=b64,
            ))
        return docs

    def _retrieve_tables(self, query: str, top_k: int) -> list[RetrievedDoc]:
        if not self._table_docs:
            return []

        tbl_emb = self.embedder.encode(
            [d["content"] for d in self._table_docs], convert_to_numpy=True
        )
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        scores = (tbl_emb @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievedDoc(
                doc_id=self._table_docs[i]["id"],
                content=self._table_docs[i]["content"],
                modality="table",
                score=float(scores[i]),
                metadata=self._table_docs[i].get("metadata", {}),
            )
            for i in top_idx
        ]
