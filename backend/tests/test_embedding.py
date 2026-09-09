from app.services.embedding_service import EmbeddingService


def test_embedding_is_deterministic():
    service = EmbeddingService()
    first = service.embed("caries risk lower molar")
    second = service.embed("caries risk lower molar")
    assert first == second
    assert len(first) == service.dim

