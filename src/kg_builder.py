"""
Step 3: Dynamic Local Knowledge Graph Construction
논문 3.3절: 검색된 문서에서 패턴 그래프를 스키마로 삼아 로컬 KG를 동적 생성
"""

import json
import re
from dataclasses import dataclass, field

import anthropic
import networkx as nx

from .query_analyzer import PatternGraph
from .retriever import RetrievedDoc


@dataclass
class KGNode:
    node_id: str
    entity_type: str
    name: str
    modality: str = "text"
    attributes: dict = field(default_factory=dict)


@dataclass
class KGEdge:
    src_id: str
    dst_id: str
    relation: str
    weight: float = 1.0
    source_doc: str = ""


class LocalKGBuilder:
    """
    검색된 문서에서 로컬 지식 그래프를 동적으로 구축.
    패턴 그래프를 스키마로 사용해 관련 트리플만 추출.
    """

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client = client
        self.model = model

    def build(
        self,
        retrieved_docs: list[RetrievedDoc],
        pattern: PatternGraph,
    ) -> nx.DiGraph:
        graph = nx.DiGraph()

        for doc in retrieved_docs:
            if doc.modality == "text":
                triples = self._extract_triples_from_text(doc, pattern)
            elif doc.modality == "table":
                triples = self._extract_triples_from_table(doc, pattern)
            elif doc.modality == "image":
                triples = self._extract_triples_from_image(doc, pattern)
            else:
                triples = []

            for triple in triples:
                self._add_triple(graph, triple, doc.doc_id)

        print(f"[KGBuilder] 노드 {graph.number_of_nodes()}개, 엣지 {graph.number_of_edges()}개 생성")
        return graph

    # ------------------------------------------------------------------ #
    # 트리플 추출
    # ------------------------------------------------------------------ #
    def _extract_triples_from_text(
        self, doc: RetrievedDoc, pattern: PatternGraph
    ) -> list[dict]:
        prompt = f"""Extract knowledge graph triples from the text below.
Focus on entity types: {pattern.entity_types}
Focus on relation types: {pattern.relation_types}
Core entity of interest: "{pattern.focus_entity}"

Text:
\"\"\"
{doc.content[:1500]}
\"\"\"

Return a JSON array of triples:
[{{"subject": "...", "subject_type": "...", "predicate": "...", "object": "...", "object_type": "..."}}]

Only include triples relevant to answering questions about "{pattern.focus_entity}".
Return ONLY the JSON array."""

        return self._call_llm_for_triples(prompt)

    def _extract_triples_from_table(
        self, doc: RetrievedDoc, pattern: PatternGraph
    ) -> list[dict]:
        prompt = f"""Extract knowledge graph triples from this table.
Core entity: "{pattern.focus_entity}"
Relation types to focus on: {pattern.relation_types}

Table:
{doc.content[:1500]}

Return a JSON array:
[{{"subject": "...", "subject_type": "...", "predicate": "...", "object": "...", "object_type": "..."}}]
Return ONLY the JSON array."""

        return self._call_llm_for_triples(prompt)

    def _extract_triples_from_image(
        self, doc: RetrievedDoc, pattern: PatternGraph
    ) -> list[dict]:
        if not doc.image_b64:
            # 이미지 파일 없으면 캡션으로 대체
            return self._extract_triples_from_text(
                RetrievedDoc(
                    doc_id=doc.doc_id,
                    content=f"[Image caption] {doc.content}",
                    modality="text",
                    score=doc.score,
                    metadata=doc.metadata,
                ),
                pattern,
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": doc.image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"""Extract knowledge graph triples from this image.
Core entity: "{pattern.focus_entity}"
Entity types: {pattern.entity_types}
Relation types: {pattern.relation_types}

Return a JSON array:
[{{"subject": "...", "subject_type": "...", "predicate": "...", "object": "...", "object_type": "..."}}]
Return ONLY the JSON array.""",
                    },
                ],
            }
        ]

        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=512, messages=messages
            )
            return self._parse_triples(response.content[0].text)
        except Exception as e:
            print(f"[KGBuilder] 이미지 트리플 추출 실패: {e}")
            return []

    def _call_llm_for_triples(self, prompt: str) -> list[dict]:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_triples(response.content[0].text)
        except Exception as e:
            print(f"[KGBuilder] 트리플 추출 실패: {e}")
            return []

    @staticmethod
    def _parse_triples(raw: str) -> list[dict]:
        raw = re.sub(r"^```json\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------ #
    # 그래프 추가
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_triple(graph: nx.DiGraph, triple: dict, source_doc: str) -> None:
        subj = triple.get("subject", "").strip()
        pred = triple.get("predicate", "").strip()
        obj = triple.get("object", "").strip()
        if not (subj and pred and obj):
            return

        if subj not in graph:
            graph.add_node(
                subj,
                entity_type=triple.get("subject_type", "Unknown"),
                modality="text",
            )
        if obj not in graph:
            graph.add_node(
                obj,
                entity_type=triple.get("object_type", "Unknown"),
                modality="text",
            )
        graph.add_edge(subj, obj, relation=pred, source_doc=source_doc)
