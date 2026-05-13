"""
Step 5: Graph-based Reasoning & Answer Generation
논문 3.5절: 로컬 KG를 순회하며 Claude로 최종 답변 생성
"""

import json
from dataclasses import dataclass

import anthropic
import networkx as nx

from .query_analyzer import PatternGraph


@dataclass
class ReasoningResult:
    answer: str
    reasoning_chain: list[str]
    evidence_nodes: list[str]
    evidence_edges: list[tuple]
    confidence: float


class GraphReasoner:
    """
    구축된 로컬 KG 위에서 멀티홉 추론을 수행하고 최종 답변을 생성.
    """

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client = client
        self.model = model

    def reason(self, graph: nx.DiGraph, pattern: PatternGraph) -> ReasoningResult:
        # 1. 핵심 서브그래프 추출
        subgraph_text = self._serialize_graph(graph, pattern)

        # 2. 멀티홉 경로 탐색
        paths = self._find_relevant_paths(graph, pattern)

        # 3. Claude로 최종 추론
        answer_data = self._generate_answer(
            query=pattern.raw_query,
            graph_text=subgraph_text,
            paths=paths,
            pattern=pattern,
        )

        return ReasoningResult(
            answer=answer_data.get("answer", ""),
            reasoning_chain=answer_data.get("reasoning_chain", []),
            evidence_nodes=answer_data.get("evidence_nodes", []),
            evidence_edges=[(e[0], e[1]) for e in paths[:5]],
            confidence=answer_data.get("confidence", 0.0),
        )

    # ------------------------------------------------------------------ #
    # 그래프 직렬화
    # ------------------------------------------------------------------ #
    def _serialize_graph(self, graph: nx.DiGraph, pattern: PatternGraph) -> str:
        lines = ["[Knowledge Graph Triples]"]

        focus = pattern.focus_entity.lower()
        relevant_nodes = set()

        # 핵심 엔티티와 연결된 노드 우선 수집
        for node in graph.nodes():
            if focus and focus in node.lower():
                relevant_nodes.add(node)
                relevant_nodes.update(graph.successors(node))
                relevant_nodes.update(graph.predecessors(node))

        if not relevant_nodes:
            relevant_nodes = set(graph.nodes())

        count = 0
        for u, v, data in graph.edges(data=True):
            if u in relevant_nodes or v in relevant_nodes:
                rel = data.get("relation", "related_to")
                lines.append(f"  ({u}) --[{rel}]--> ({v})")
                count += 1
                if count >= 40:  # 토큰 절약
                    lines.append("  ... (더 많은 트리플 존재)")
                    break

        # 노드 속성 (시각적 설명, 테이블 등)
        lines.append("\n[Node Attributes]")
        for node in relevant_nodes:
            attrs = graph.nodes[node]
            if "visual_description" in attrs:
                lines.append(f"  {node} [image]: {attrs['visual_description']}")
            if "table_data" in attrs:
                lines.append(f"  {node} [table]: {attrs['table_data'][:200]}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 멀티홉 경로 탐색
    # ------------------------------------------------------------------ #
    def _find_relevant_paths(
        self, graph: nx.DiGraph, pattern: PatternGraph
    ) -> list[list]:
        paths = []
        focus = pattern.focus_entity.lower()

        start_nodes = [
            n for n in graph.nodes() if focus and focus in n.lower()
        ]
        if not start_nodes:
            start_nodes = list(graph.nodes())[:3]

        for start in start_nodes[:2]:
            for end in graph.nodes():
                if start == end:
                    continue
                try:
                    path = nx.shortest_path(graph, start, end)
                    if 1 < len(path) <= pattern.hop_count + 1:
                        paths.append(path)
                except nx.NetworkXNoPath:
                    pass

        return paths[:10]

    # ------------------------------------------------------------------ #
    # 최종 답변 생성
    # ------------------------------------------------------------------ #
    def _generate_answer(
        self,
        query: str,
        graph_text: str,
        paths: list[list],
        pattern: PatternGraph,
    ) -> dict:
        path_text = ""
        if paths:
            path_text = "\n[Reasoning Paths]\n"
            for path in paths[:5]:
                path_text += "  " + " → ".join(path) + "\n"

        multimodal_hint = ""
        if "image" in pattern.required_modalities:
            multimodal_hint = "\nNote: Visual information from images is included in node attributes."
        if "table" in pattern.required_modalities:
            multimodal_hint += "\nNote: Tabular data is included in node attributes."

        prompt = f"""You are a multimodal knowledge graph reasoning expert.

Use the knowledge graph below to answer the question.

Question: "{query}"

{graph_text}
{path_text}
{multimodal_hint}

Provide your answer as a JSON object:
{{
  "answer": "<concise final answer>",
  "reasoning_chain": ["<step 1>", "<step 2>", "..."],
  "evidence_nodes": ["<key node 1>", "<key node 2>", "..."],
  "confidence": <0.0 to 1.0>
}}

- reasoning_chain: step-by-step reasoning using the KG triples
- evidence_nodes: the most important nodes used to reach the answer
- confidence: how confident you are (1.0 = certain, 0.0 = guessing)

Return ONLY the JSON."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = raw.lstrip("```json").rstrip("```").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"[Reasoner] 답변 생성 실패: {e}")
            return {
                "answer": "답변을 생성할 수 없습니다.",
                "reasoning_chain": [],
                "evidence_nodes": [],
                "confidence": 0.0,
            }
