"""
Step 1: Query Semantic Analysis & Pattern Graph Construction
논문의 핵심: 쿼리 의미에서 그래프 패턴(엔티티 타입 + 관계 타입)을 추출
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic


@dataclass
class PatternGraph:
    """쿼리로부터 추출한 패턴 그래프 (KG 구성 스키마)"""
    question_type: str           # what / who / when / where / how / yes-no
    required_modalities: list[str]  # text / image / table
    entity_types: list[str]      # 기대하는 엔티티 타입
    relation_types: list[str]    # 기대하는 관계 타입
    hop_count: int               # 추론에 필요한 hop 수
    focus_entity: str            # 쿼리의 핵심 엔티티
    subqueries: list[str]        # 복잡한 쿼리를 분해한 서브쿼리 목록
    raw_query: str = ""


class QueryAnalyzer:
    """
    쿼리를 분석해 패턴 그래프를 생성.
    논문 3.1절: Pattern Graph Construction from Query Semantics
    """

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client = client
        self.model = model

    def analyze(self, query: str) -> PatternGraph:
        prompt = f"""You are an expert at analyzing questions for a multimodal knowledge graph RAG system.

Analyze the following question and extract a structured pattern graph that will guide knowledge retrieval.

Question: "{query}"

Return a JSON object with exactly these fields:
{{
  "question_type": "<what|who|when|where|how|yes_no|count>",
  "required_modalities": ["<text|image|table>"],
  "entity_types": ["<list of entity types needed, e.g. Person, Organization, Location, Event, Product>"],
  "relation_types": ["<list of relation types, e.g. born_in, founded_by, located_in, part_of>"],
  "hop_count": <integer 1-3>,
  "focus_entity": "<the main entity the question is about>",
  "subqueries": ["<decomposed sub-questions if multi-hop, else just the original>"]
}}

Rules:
- required_modalities: include "image" if question asks about visual appearance, "table" if about statistics/numbers/comparisons
- hop_count: 1 for direct questions, 2+ for questions requiring bridging entities
- subqueries: for multi-hop questions, break into single-hop steps

Return ONLY the JSON, no explanation."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data: dict[str, Any] = json.loads(raw)

        return PatternGraph(
            question_type=data.get("question_type", "what"),
            required_modalities=data.get("required_modalities", ["text"]),
            entity_types=data.get("entity_types", []),
            relation_types=data.get("relation_types", []),
            hop_count=data.get("hop_count", 1),
            focus_entity=data.get("focus_entity", ""),
            subqueries=data.get("subqueries", [query]),
            raw_query=query,
        )
