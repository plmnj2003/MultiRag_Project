"""
MultimodalQA 데이터셋 로더
https://github.com/allenai/multimodalqa
"""

import json
from pathlib import Path


def load_multimodalqa_sample(data_dir: str = "./data/multimodalqa") -> dict:
    """
    MultimodalQA 데이터셋에서 샘플을 로드.
    data_dir에 파일이 없으면 내장 샘플 데이터를 반환.
    """
    data_path = Path(data_dir) / "MMQA_dev.jsonl"

    if data_path.exists():
        return _load_from_file(str(data_path))
    else:
        print("[DataLoader] MultimodalQA 파일 없음 → 내장 샘플 데이터 사용")
        return _get_builtin_samples()


def _load_from_file(path: str, max_samples: int = 20) -> dict:
    samples = {"questions": [], "texts": {}, "tables": {}, "images": {}}

    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            item = json.loads(line)
            samples["questions"].append({
                "qid": item.get("qid", f"q{i}"),
                "question": item.get("question", ""),
                "answer": item.get("answers", [{}])[0].get("answer", ""),
                "modalities": item.get("metadata", {}).get("modalities", ["text"]),
            })

    print(f"[DataLoader] {len(samples['questions'])}개 질문 로드 완료")
    return samples


def _get_builtin_samples() -> dict:
    """논문 실습용 내장 샘플 데이터"""
    texts = [
        {
            "id": "t001",
            "content": "Albert Einstein was born on March 14, 1879, in Ulm, Kingdom of Württemberg, German Empire. He developed the theory of relativity, one of the two pillars of modern physics. Einstein is best known for his mass–energy equivalence formula E = mc².",
            "metadata": {"source": "wikipedia", "topic": "physics"},
        },
        {
            "id": "t002",
            "content": "The Nobel Prize in Physics 1921 was awarded to Albert Einstein 'for his services to Theoretical Physics, and especially for his discovery of the law of the photoelectric effect'. Einstein received the prize in 1922.",
            "metadata": {"source": "nobelprize.org", "topic": "awards"},
        },
        {
            "id": "t003",
            "content": "Marie Curie was a Polish and naturalised-French physicist and chemist. She was the first woman to win a Nobel Prize, the first person to win the Nobel Prize twice, and the only person to win the Nobel Prize in two scientific sciences. She won the Physics prize in 1903 and Chemistry prize in 1911.",
            "metadata": {"source": "wikipedia", "topic": "physics"},
        },
        {
            "id": "t004",
            "content": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. The tower is 330 metres (1,083 ft) tall.",
            "metadata": {"source": "wikipedia", "topic": "architecture"},
        },
        {
            "id": "t005",
            "content": "Gustave Eiffel was a French civil engineer. A graduate of École Centrale des Arts et Manufactures, he made his name with various bridges for the French railway network. He is best known for the world-famous Eiffel Tower, designed for the 1889 Universal Exposition in Paris.",
            "metadata": {"source": "wikipedia", "topic": "engineering"},
        },
        {
            "id": "t006",
            "content": "The 1889 Universal Exposition (Exposition Universelle) was a World's Fair held in Paris, France from May 5 to October 31, 1889. It was held to celebrate the centennial of the French Revolution. The Eiffel Tower was built as the entrance arch for the exposition.",
            "metadata": {"source": "wikipedia", "topic": "history"},
        },
        {
            "id": "t007",
            "content": "The Louvre Museum in Paris is the world's largest art museum and a historic monument. It is home to approximately 35,000 works of art. The iconic glass pyramid entrance was designed by architect I. M. Pei and completed in 1989.",
            "metadata": {"source": "wikipedia", "topic": "museum"},
        },
        {
            "id": "t008",
            "content": "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne. The company is headquartered in Cupertino, California. Apple designs, develops, and sells consumer electronics, computer software, and online services.",
            "metadata": {"source": "wikipedia", "topic": "technology"},
        },
        {
            "id": "t009",
            "content": "Steve Jobs was an American business magnate, inventor, and investor. He was the co-founder, chairman, and CEO of Apple Inc. Jobs was also the primary investor and CEO of Pixar. He died on October 5, 2011, in Palo Alto, California.",
            "metadata": {"source": "wikipedia", "topic": "technology"},
        },
        {
            "id": "t010",
            "content": "Pixar Animation Studios is an American computer animation film studio. It was acquired by The Walt Disney Company in 2006. Pixar has produced famous films including Toy Story (1995), Finding Nemo (2003), The Incredibles (2004), and WALL-E (2008).",
            "metadata": {"source": "wikipedia", "topic": "entertainment"},
        },
    ]

    tables = [
        {
            "id": "tbl001",
            "content": """| Nobel Prize Winners in Physics (Selected) |
|------|------|------|
| Year | Laureate | Achievement |
| 1903 | Marie Curie | Research on radiation phenomena |
| 1921 | Albert Einstein | Photoelectric effect |
| 1965 | Richard Feynman | Quantum electrodynamics |
| 1984 | Carlo Rubbia | Discovery of W and Z particles |""",
            "metadata": {"source": "nobelprize.org"},
        },
        {
            "id": "tbl002",
            "content": """| Apple Inc. Financial Highlights |
|------|------|
| Year | Revenue (USD Billions) |
| 2020 | 274.5 |
| 2021 | 365.8 |
| 2022 | 394.3 |
| 2023 | 383.3 |""",
            "metadata": {"source": "apple.com/investor"},
        },
        {
            "id": "tbl003",
            "content": """| Eiffel Tower Facts |
|------|------|
| Attribute | Value |
| Height | 330 metres (1,083 ft) |
| Construction start | January 28, 1887 |
| Construction end | March 31, 1889 |
| Designer | Gustave Eiffel |
| Location | Paris, France |
| Visitors per year | ~7 million |""",
            "metadata": {"source": "eiffel-tower.com"},
        },
    ]

    questions = [
        {
            "qid": "q001",
            "question": "What award did Albert Einstein receive for his discovery of the photoelectric effect, and in which year?",
            "answer": "Nobel Prize in Physics, 1921",
            "modalities": ["text", "table"],
        },
        {
            "qid": "q002",
            "question": "Who designed the Eiffel Tower and what was the occasion for which it was built?",
            "answer": "Gustave Eiffel designed it for the 1889 Universal Exposition in Paris",
            "modalities": ["text"],
        },
        {
            "qid": "q003",
            "question": "Who was the first woman to win a Nobel Prize, and how many Nobel Prizes did she win in total?",
            "answer": "Marie Curie, she won two Nobel Prizes",
            "modalities": ["text", "table"],
        },
        {
            "qid": "q004",
            "question": "What company did Steve Jobs co-found, and what animation studio did he later invest in?",
            "answer": "Steve Jobs co-founded Apple Inc. and was the primary investor of Pixar",
            "modalities": ["text"],
        },
        {
            "qid": "q005",
            "question": "How tall is the Eiffel Tower and how many visitors does it receive per year?",
            "answer": "330 metres tall, approximately 7 million visitors per year",
            "modalities": ["text", "table"],
        },
    ]

    return {
        "questions": questions,
        "texts": {d["id"]: d for d in texts},
        "tables": {d["id"]: d for d in tables},
        "images": {},
    }
