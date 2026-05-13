"""
Query-Driven Multimodal GraphRAG - 전체 파이프라인
논문: "Query-Driven Multimodal GraphRAG: Dynamic Local Knowledge Graph Construction for Online Reasoning"
ACL 2025 Findings
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import anthropic
import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv

from .kg_builder import LocalKGBuilder
from .multimodal_supplement import MultimodalSupplementer
from .query_analyzer import PatternGraph, QueryAnalyzer
from .reasoner import GraphReasoner, ReasoningResult
from .retriever import MultiPathRetriever

load_dotenv()


@dataclass
class PipelineOutput:
    query: str
    pattern: PatternGraph
    graph: nx.DiGraph
    result: ReasoningResult


class MultimodalGraphRAG:
    """
    Query-Driven Multimodal GraphRAG 전체 파이프라인.

    5단계:
    1. Query Analysis     → Pattern Graph
    2. Multi-path Retrieval
    3. Local KG Construction
    4. Multimodal Supplement
    5. Graph-based Reasoning
    """

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        self.query_analyzer = QueryAnalyzer(self.client, model)
        self.retriever = MultiPathRetriever(embedding_model)
        self.kg_builder = LocalKGBuilder(self.client, model)
        self.supplementer = MultimodalSupplementer(self.client, model, self.retriever)
        self.reasoner = GraphReasoner(self.client, model)

    # ------------------------------------------------------------------ #
    # 데이터 로딩
    # ------------------------------------------------------------------ #
    def load_texts(self, docs: list[dict]) -> None:
        self.retriever.index_texts(docs)

    def load_images(self, docs: list[dict]) -> None:
        self.retriever.index_images(docs)

    def load_tables(self, docs: list[dict]) -> None:
        self.retriever.index_tables(docs)

    # ------------------------------------------------------------------ #
    # 메인 파이프라인
    # ------------------------------------------------------------------ #
    def run(self, query: str, top_k: int = 5, verbose: bool = True) -> PipelineOutput:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)

        # Step 1: Query Analysis
        if verbose:
            print("\n[Step 1] Query 분석 → Pattern Graph 생성...")
        pattern = self.query_analyzer.analyze(query)
        if verbose:
            print(f"  - 질문 유형: {pattern.question_type}")
            print(f"  - 필요 모달: {pattern.required_modalities}")
            print(f"  - 핵심 엔티티: {pattern.focus_entity}")
            print(f"  - 서브쿼리: {pattern.subqueries}")
            print(f"  - 추론 hop: {pattern.hop_count}")

        # Step 2: Multi-path Retrieval
        if verbose:
            print("\n[Step 2] 다중 경로 검색...")
        docs = self.retriever.retrieve(pattern, top_k=top_k)
        if verbose:
            for d in docs:
                print(f"  [{d.modality.upper()}] (score={d.score:.3f}) {d.doc_id}: {d.content[:80]}...")

        # Step 3: Local KG Construction
        if verbose:
            print("\n[Step 3] 로컬 지식 그래프 구축...")
        graph = self.kg_builder.build(docs, pattern)

        # Step 4: Multimodal Supplement
        if verbose:
            print("\n[Step 4] 멀티모달 정보 보충...")
        graph = self.supplementer.supplement(graph, pattern)

        # Step 5: Reasoning
        if verbose:
            print("\n[Step 5] 그래프 기반 추론...")
        result = self.reasoner.reason(graph, pattern)

        if verbose:
            print(f"\n{'='*60}")
            print(f"최종 답변: {result.answer}")
            print(f"신뢰도: {result.confidence:.2f}")
            print(f"\n추론 과정:")
            for i, step in enumerate(result.reasoning_chain, 1):
                print(f"  {i}. {step}")
            print(f"\n근거 노드: {result.evidence_nodes}")
            print('='*60)

        return PipelineOutput(
            query=query,
            pattern=pattern,
            graph=graph,
            result=result,
        )

    # ------------------------------------------------------------------ #
    # 시각화
    # ------------------------------------------------------------------ #
    def visualize_graph(
        self,
        output: PipelineOutput,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> None:
        graph = output.graph
        if graph.number_of_nodes() == 0:
            print("그래프가 비어있습니다.")
            return

        plt.figure(figsize=figsize)

        pos = nx.spring_layout(graph, seed=42, k=2)

        # 엔티티 타입별 색상
        node_colors = []
        evidence_set = set(output.result.evidence_nodes)
        for node in graph.nodes():
            if node in evidence_set:
                node_colors.append("#FF6B6B")
            else:
                node_colors.append("#4ECDC4")

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=800, alpha=0.9)
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")

        edge_labels = {(u, v): d.get("relation", "") for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=15, edge_color="#888888", alpha=0.7)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

        plt.title(f"Local Knowledge Graph\nQuery: {output.query[:60]}", fontsize=12)
        plt.figtext(
            0.01, 0.01,
            f"Answer: {output.result.answer[:80]}",
            wrap=True, fontsize=9, color="#333333"
        )
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[시각화] 저장 완료: {save_path}")
        else:
            plt.show()
        plt.close()
