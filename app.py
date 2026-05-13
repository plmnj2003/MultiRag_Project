"""
Multimodal GraphRAG 웹 실습 인터페이스
"""

import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)

rag = None
dataset = None


def get_pipeline():
    global rag, dataset
    if rag is None:
        from src.data_loader import load_multimodalqa_sample
        from src.pipeline import MultimodalGraphRAG

        dataset = load_multimodalqa_sample()
        rag = MultimodalGraphRAG()
        rag.load_texts(list(dataset["texts"].values()))
        rag.load_tables(list(dataset["tables"].values()))
    return rag, dataset


def graph_to_base64(graph, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    if graph.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42, k=2)
    evidence_set = set(result.evidence_nodes)
    node_colors = ["#FF6B6B" if n in evidence_set else "#4ECDC4" for n in graph.nodes()]

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=900, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold", ax=ax)
    edge_labels = {(u, v): d.get("relation", "") for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowsize=20, edge_color="#888", alpha=0.7, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.axis("off")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    result_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return result_b64


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/questions")
def get_questions():
    _, ds = get_pipeline()
    questions = [{"qid": q["qid"], "question": q["question"], "answer": q["answer"]} for q in ds["questions"]]
    return jsonify(questions)


@app.route("/api/run", methods=["POST"])
def run_query():
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "질문을 입력해주세요."}), 400

    def generate():
        def send(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            # 초기화
            yield send({"step": "init", "status": "running", "message": "파이프라인 초기화 중..."})
            pipeline, _ = get_pipeline()
            yield send({"step": "init", "status": "done", "message": "초기화 완료"})

            # Step 1: Query Analysis
            yield send({"step": 1, "status": "running"})
            pattern = pipeline.query_analyzer.analyze(query)
            yield send({
                "step": 1, "status": "done",
                "data": {
                    "question_type": pattern.question_type,
                    "required_modalities": pattern.required_modalities,
                    "focus_entity": pattern.focus_entity,
                    "subqueries": pattern.subqueries,
                    "hop_count": pattern.hop_count,
                    "entity_types": pattern.entity_types,
                    "relation_types": pattern.relation_types,
                }
            })

            # Step 2: Multi-path Retrieval
            yield send({"step": 2, "status": "running"})
            docs = pipeline.retriever.retrieve(pattern, top_k=5)
            yield send({
                "step": 2, "status": "done",
                "data": [
                    {"modality": d.modality, "doc_id": d.doc_id, "score": round(d.score, 3), "content": d.content[:120]}
                    for d in docs
                ]
            })

            # Step 3: Local KG Construction
            yield send({"step": 3, "status": "running"})
            graph = pipeline.kg_builder.build(docs, pattern)
            yield send({
                "step": 3, "status": "done",
                "data": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}
            })

            # Step 4: Multimodal Supplement
            yield send({"step": 4, "status": "running"})
            graph = pipeline.supplementer.supplement(graph, pattern)
            yield send({
                "step": 4, "status": "done",
                "data": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}
            })

            # Step 5: Graph Reasoning
            yield send({"step": 5, "status": "running"})
            result = pipeline.reasoner.reason(graph, pattern)
            graph_img = graph_to_base64(graph, result)
            yield send({
                "step": 5, "status": "done",
                "data": {
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "reasoning_chain": result.reasoning_chain,
                    "evidence_nodes": result.evidence_nodes,
                    "graph_image": graph_img,
                }
            })

            yield send({"step": "complete"})

        except Exception as e:
            yield send({"step": "error", "message": str(e)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
