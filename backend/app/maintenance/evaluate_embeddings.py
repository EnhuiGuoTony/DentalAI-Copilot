"""小规模合成资料评估：比较旧哈希基线与指定语义模型，不读写数据库或患者数据。

python -m app.maintenance.evaluate_embeddings --live 会调用配置的真实提供方。
报告 top-1 命中率与 MRR；只验证这些例子，不能视为医学质量或整体召回保证。
"""
import argparse
import json

from app.services.embedding_service import EmbeddingService

DOCUMENTS = [
    "The patient reports bleeding gums while brushing teeth.",
    "A filling was placed in the lower left molar to restore a cavity.",
    "No known drug allergies were reported.",
    "The next dental cleaning is scheduled for next month.",
    "患者刷牙时牙龈出血，已建议复诊检查。",
    "患者左下磨牙遇冷敏感，咀嚼时疼痛。",
]
CASES = [
    ("Gingival hemorrhage during oral hygiene", 0),
    ("Restoration of tooth decay on the bottom left", 1),
    ("Any adverse reactions to medications?", 2),
    ("When is the upcoming hygiene visit?", 3),
    ("有没有牙龈流血的记录？", 4),
    ("哪条记录提到吃冷东西牙齿不舒服？", 5),
]


def metrics(documents: list[list[float]], queries: list[list[float]]) -> dict:
    """向量已归一化，用内积得到余弦相似度；相同分数按原始文档顺序排序。"""
    ranks = []
    for query, (_, expected) in zip(queries, CASES):
        scores = [sum(a * b for a, b in zip(query, document)) for document in documents]
        ordered = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        ranks.append(ordered.index(expected) + 1)
    return {"top1": sum(rank == 1 for rank in ranks) / len(ranks),
            "mrr": sum(1 / rank for rank in ranks) / len(ranks), "ranks": ranks}


def main() -> None:
    """显式 --live 才调用远端；只打印指标，不打印密钥、数据库或患者内容。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    service = EmbeddingService()
    result = {"cases": len(CASES), "hash_baseline": metrics(
        [service._hash_embed(document) for document in DOCUMENTS],
        [service._hash_embed(query) for query, _ in CASES])}
    if args.live:
        if service.settings.embedding_provider != "openrouter":
            raise ValueError("Live evaluation requires EMBEDDING_PROVIDER=openrouter")
        result["model"] = service.settings.embedding_model
        result["semantic"] = metrics(service.embed_documents(DOCUMENTS),
                                     [service.embed_query(query) for query, _ in CASES])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
