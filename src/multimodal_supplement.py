"""
Step 4: Multimodal Information Supplementation
논문 3.4절: KG에서 누락된 멀티모달 정보를 보완 (이미지/테이블 ad hoc 보충)
"""

import anthropic
import networkx as nx

from .query_analyzer import PatternGraph
from .retriever import MultiPathRetriever


class MultimodalSupplementer:
    """
    구축된 KG에서 멀티모달 정보가 누락된 노드를 탐지하고 보충.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        retriever: MultiPathRetriever,
    ):
        self.client = client
        self.model = model
        self.retriever = retriever

    def supplement(self, graph: nx.DiGraph, pattern: PatternGraph) -> nx.DiGraph:
        if "image" not in pattern.required_modalities and "table" not in pattern.required_modalities:
            return graph

        nodes_needing_visual = [
            n for n, data in graph.nodes(data=True)
            if data.get("modality") == "text" and data.get("entity_type") in (
                "Person", "Product", "Building", "Animal", "Artwork"
            )
        ]

        for node_name in nodes_needing_visual[:3]:  # 최대 3개만 보충
            if "image" in pattern.required_modalities:
                self._add_image_info(graph, node_name)
            if "table" in pattern.required_modalities:
                self._add_table_info(graph, node_name)

        return graph

    def _add_image_info(self, graph: nx.DiGraph, entity: str) -> None:
        """이미지 검색 후 시각적 설명을 노드 속성에 추가"""
        image_docs = self.retriever._retrieve_images(entity, top_k=1)
        if not image_docs:
            return

        doc = image_docs[0]
        if doc.image_b64:
            desc = self._describe_image(doc.image_b64, entity)
            graph.nodes[entity]["visual_description"] = desc
            graph.nodes[entity]["modality"] = "multimodal"
            print(f"[Supplement] '{entity}' 이미지 보충 완료")
        elif doc.content:
            graph.nodes[entity]["visual_description"] = doc.content
            graph.nodes[entity]["modality"] = "multimodal"

    def _add_table_info(self, graph: nx.DiGraph, entity: str) -> None:
        """테이블 검색 후 수치 정보를 노드 속성에 추가"""
        table_docs = self.retriever._retrieve_tables(entity, top_k=1)
        if not table_docs:
            return

        doc = table_docs[0]
        graph.nodes[entity]["table_data"] = doc.content[:500]
        graph.nodes[entity]["modality"] = "multimodal"
        print(f"[Supplement] '{entity}' 테이블 보충 완료")

    def _describe_image(self, image_b64: str, entity: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Describe what you see in this image about '{entity}' in 1-2 sentences.",
                        },
                    ],
                }],
            )
            return response.content[0].text
        except Exception as e:
            return f"[이미지 설명 실패: {e}]"
