"""
Query-Driven Multimodal GraphRAG - 메인 실행 파일
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_multimodalqa_sample
from src.pipeline import MultimodalGraphRAG


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        print("ERROR: .env 파일에 ANTHROPIC_API_KEY를 설정해주세요.")
        sys.exit(1)

    # 데이터 로드
    print("데이터 로딩...")
    dataset = load_multimodalqa_sample()

    # 파이프라인 초기화
    print("파이프라인 초기화...")
    rag = MultimodalGraphRAG()

    # 문서 인덱싱
    texts = list(dataset["texts"].values())
    tables = list(dataset["tables"].values())
    rag.load_texts(texts)
    rag.load_tables(tables)

    # 출력 디렉토리
    Path("outputs").mkdir(exist_ok=True)

    # 실행할 질문 (멀티홉 포함)
    questions = dataset["questions"]

    print(f"\n총 {len(questions)}개 질문 처리\n")

    correct = 0
    for i, q_item in enumerate(questions):
        query = q_item["question"]
        expected = q_item["answer"]

        output = rag.run(query, top_k=5, verbose=True)

        # 그래프 시각화 저장
        viz_path = f"outputs/graph_q{i+1:02d}.png"
        rag.visualize_graph(output, save_path=viz_path)

        # 정답 비교 (간단한 키워드 매칭)
        answer_lower = output.result.answer.lower()
        expected_lower = expected.lower()
        is_correct = any(
            kw in answer_lower for kw in expected_lower.split() if len(kw) > 3
        )
        if is_correct:
            correct += 1

        print(f"\n기대 답변: {expected}")
        print(f"예측 답변: {output.result.answer}")
        print(f"정답 여부: {'✓' if is_correct else '✗'}")
        print("-" * 60)

    accuracy = correct / len(questions) * 100
    print(f"\n{'='*60}")
    print(f"최종 정확도: {correct}/{len(questions)} ({accuracy:.1f}%)")
    print(f"시각화 결과: outputs/ 폴더 확인")
    print('='*60)


if __name__ == "__main__":
    main()
