"""向量接口测试使用本地 HTTP transport，验证真实 SDK 请求但不访问远端。"""
import json
import math

import httpx
import pytest
from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings, Settings
from app.services.embedding_service import EmbeddingService, EmbeddingError
from app.services.chunking import chunk_text, byte_length


def test_embedding_is_deterministic(monkeypatch):
    monkeypatch.setattr(get_settings(), "embedding_provider", "hash")
    service = EmbeddingService()
    first = service.embed("caries risk lower molar")
    second = service.embed("caries risk lower molar")
    assert first == second
    assert len(first) == service.dim


@pytest.mark.parametrize("text", ["牙龈出血，建议复诊。" * 100, "word " * 500, "牙" * 600, "🦷" * 300])
def test_splitter_bounds_and_content(text):
    """覆盖连续中文和 emoji，避免空白分词把长中文当作一个词。"""
    chunks = chunk_text(text, overlap=0)
    assert len(chunks) > 1
    assert all(0 < byte_length(chunk) <= 480 for chunk in chunks)
    assert "".join("".join(chunks).split()) == "".join(text.split())


def test_splitter_boundaries_overlap_and_validation():
    assert chunk_text(" \n ") == []
    text = "First paragraph.\n\nSecond paragraph."
    assert chunk_text(text, chunk_size=20, overlap=0) == ["First paragraph.", "Second paragraph."]
    assert chunk_text("a" * 30, chunk_size=20, overlap=5) == ["a" * 20, "a" * 15]
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        Settings(_env_file=None, embedding_dim=384)


def install_transport(service, handler):
    """用受控 HTTP 响应替代远端，保留 LangChain 和 OpenAI SDK 的实际序列化。"""
    transport = httpx.Client(transport=httpx.MockTransport(handler))
    service.client = OpenAIEmbeddings(
        model=service.settings.embedding_model, api_key="synthetic-test-key",
        base_url="https://example.invalid/api/v1", check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"}, http_client=transport,
        max_retries=0, chunk_size=2,
    )
    return transport


def test_openrouter_wire_format_batching_and_query(monkeypatch):
    monkeypatch.setattr(get_settings(), "embedding_provider", "openrouter")
    service = EmbeddingService()
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/api/v1/embeddings"
        assert body["model"] == "liquid/lfm-2.5-embedding-350m:free"
        assert body["encoding_format"] == "float"
        assert all(isinstance(item, str) for item in body["input"])
        return httpx.Response(200, json={"data": [
            {"embedding": [1.0] + [0.0] * (service.dim - 1), "index": i}
            for i, _ in enumerate(body["input"])]})

    with install_transport(service, handler):
        assert len(service.embed_documents(["one", "two", "三"])) == 3
        assert len(service.embed_query("牙龈出血")) == 1024
    assert [len(r["input"]) for r in requests] == [2, 1, 1]


@pytest.mark.parametrize("kind", ["wrong_dimension", "empty", "zero", "nan", "count"])
def test_bad_provider_vectors_rejected(kind):
    service = EmbeddingService()
    vector = [1.0] * service.dim
    vectors = [vector]
    if kind == "wrong_dimension":
        vectors = [[1.0] * 384]
    elif kind == "empty":
        vectors = [[]]
    elif kind == "zero":
        vectors = [[0.0] * service.dim]
    elif kind == "nan":
        vector[0] = float("nan")
    else:
        vectors = []
    with pytest.raises(EmbeddingError, match="invalid"):
        service._validate(vectors, 1)


def test_failure_has_no_hash_fallback_or_sensitive_body(monkeypatch):
    monkeypatch.setattr(get_settings(), "embedding_provider", "openrouter")
    service = EmbeddingService()
    with install_transport(service, lambda _: httpx.Response(429, json={"error": {"message": "sensitive-input"}})):
        with pytest.raises(EmbeddingError) as error:
            service.embed("synthetic note")
    assert "sensitive-input" not in str(error.value)
    assert "provider request failed" in str(error.value)


def test_missing_key_and_empty_inputs(monkeypatch):
    monkeypatch.setattr(get_settings(), "embedding_provider", "openrouter")
    monkeypatch.setattr(get_settings(), "embedding_api_key", "")
    monkeypatch.setattr(get_settings(), "openrouter_api_key", "")
    service = EmbeddingService()
    with pytest.raises(EmbeddingError, match="API_KEY"):
        service.embed("synthetic")
    assert service.embed_documents([]) == []
    with pytest.raises(EmbeddingError):
        service.embed_query(" ")
    with pytest.raises(EmbeddingError):
        service.embed("x" * 481)


def test_long_query_keeps_tail(monkeypatch):
    """长问题分段后聚合，尾部内容必须到达提供方，而非被 SDK 静默截断。"""
    monkeypatch.setattr(get_settings(), "embedding_provider", "openrouter")
    service = EmbeddingService()
    captured = []

    def handler(request):
        inputs = json.loads(request.content)["input"]
        captured.extend(inputs)
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [1.] * service.dim}
                                                 for i in range(len(inputs))]})
    with install_transport(service, handler):
        vector = service.embed_query("牙" * 300 + "tail-marker")
    assert all(byte_length(item) <= 480 for item in captured)
    assert captured[-1].endswith("tail-marker")
    assert math.isclose(sum(v * v for v in vector), 1)
