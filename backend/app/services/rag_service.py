from uuid import UUID

from sqlalchemy.orm import Session

from app.schemas.rag import EvidenceItem, RagQueryResponse
from app.services.vector_store import VectorStore


class RagService:
    """病例 RAG 的演示入口：召回证据并拼接回答，目前不调用真实 LLM。

    可对照 PmsAgentService 学习区别：那里由模型选择工具，工具也可以执行 RAG；
    这里由固定代码依次执行检索、合并和回答，属于确定性工作流。
    """
    def __init__(self, db: Session) -> None:
        self.vector_store = VectorStore(db)

    def query(self, patient_id: UUID, question: str, top_k: int = 5) -> RagQueryResponse:
        """合并该患者的病历证据与全局知识，返回可展示的来源、摘要和相关性分数。"""
        patient_results = self.vector_store.search(patient_id, question, top_k)
        # 分别召回两类候选；知识候选至少取 2 条，然后共同竞争最终 top_k 个位置。
        # 这是分路召回后的分数排序，不是使用第二个模型进行语义重排（reranking）。
        knowledge_results = self.vector_store.search_knowledge(question, max(2, top_k // 2))
        results = [*patient_results, *knowledge_results]
        results.sort(key=lambda item: item[1], reverse=True)
        results = results[:top_k]
        # 截取 360 个字符用于展示，来源 UUID 和 metadata 仍保留用于定位原始记录。
        # 相同分数只表示向量评分一致，不能据此推断两条证据医学可靠性相同。
        evidence = [
            EvidenceItem(
                source_type=chunk.source_type,
                source_id=chunk.source_id,
                snippet=chunk.chunk_text[:360],
                score=round(score, 4),
                metadata=chunk.meta,
            )
            for chunk, score in results
        ]
        # 此处没有“把证据送入 LLM”的生成步骤；返回的是下面的固定模板回答。
        answer = self._compose_mock_answer(question, evidence)
        return RagQueryResponse(
            answer=answer,
            evidence=evidence,
            limitations=[
                "Evidence comes from the selected patient's indexed notes and shared dental knowledge.",
                "This endpoint returns a template summary; the conversation Agent can synthesize retrieved evidence with a chat model.",
                "AI output is for clinician review and is not a definitive diagnosis.",
            ],
        )

    @staticmethod
    def _compose_mock_answer(question: str, evidence: list[EvidenceItem]) -> str:
        """仅展示最高分证据；不综合推理，也不验证证据是否足以回答问题。"""
        if not evidence:
            return f"No indexed patient history was found for: {question}"
        top = evidence[0].snippet
        return f"Relevant patient context found for '{question}': {top}"
