"""把切片和查询交给 LangChain Embeddings，验证后返回可写入 pgvector 的向量。

真实模式失败会显式报错，绝不自动切到哈希，否则同一索引会混入不同向量空间。
此服务不提交数据库事务，不记录输入文本或上游错误正文。
"""
import hashlib
import math
from functools import cached_property

from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings
from app.services.chunking import byte_length, chunk_text


class EmbeddingError(ValueError):
    """可安全返回给 API/Agent 的配置、远端服务或向量格式错误。"""


class EmbeddingService:
    """默认调用 Liquid；hash 模式仅用于明确选择的离线教学和测试。"""

    def __init__(self) -> None:
        # 文档向量、查询向量和数据库 Vector 列必须使用相同维度与编码算法。
        # 更换算法或维度后需要重新生成历史向量；只修改配置不能完成数据迁移。
        self.settings = get_settings()
        self.dim = self.settings.embedding_dim

    @property
    def fingerprint(self) -> str:
        """标识模型、服务地址、维度及切片策略；配置变化后要求整体重建。

        密钥不参与指纹，轮换密钥不会使语义空间失效；策略版本变化则必须重建。
        """
        model = (self.settings.embedding_base_url.rstrip("/") + ":" + self.settings.embedding_model
                 if self.settings.embedding_provider == "openrouter" else "signed-sha256-v1")
        value = f"{model}:{self.dim}:recursive-utf8-v1:{self.settings.rag_chunk_size}:{self.settings.rag_chunk_overlap}"
        return hashlib.sha256(value.encode()).hexdigest()

    @cached_property
    def client(self) -> OpenAIEmbeddings:
        """首次需要真实编码时才创建客户端；预览导入和离线模式不要求密钥。

        check_embedding_ctx_length=False 保留原始字符串，不让 OpenAI tokenizer
        把 Liquid 输入变为不兼容的 token ID。HTTP 协议字段为 encoding_format，
        对应用户 JavaScript SDK 中的 encodingFormat。超时和重试由 SDK 控制。
        """
        key = self.settings.embedding_api_key or self.settings.openrouter_api_key
        if not key:
            raise EmbeddingError("Set EMBEDDING_API_KEY or OPENROUTER_API_KEY for embeddings")
        return OpenAIEmbeddings(
            model=self.settings.embedding_model, api_key=key,
            base_url=self.settings.embedding_base_url,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
            chunk_size=self.settings.embedding_batch_size,
            request_timeout=self.settings.embedding_timeout,
            max_retries=self.settings.embedding_max_retries,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码已切好的文本；全部验证成功后调用者才应写库。

        拒绝空文本和超长块，避免提供方静默截断。维度、数量、有限值及非零范数
        都是数据库余弦检索的前置条件；不会截断向量或补零伪造正确维度。
        """
        if not texts:
            return []
        if any(not text.strip() or byte_length(text) > 480 for text in texts):
            raise EmbeddingError("Embedding documents must be nonempty chunks of at most 480 UTF-8 bytes")
        if self.settings.embedding_provider == "hash":
            vectors = [self._hash_embed(text) for text in texts]
        else:
            client = self.client
            try:
                vectors = client.embed_documents(texts)
            except Exception:
                # 上游异常可能包含原文、请求参数或密钥，不能直接传递给 UI 或日志。
                raise EmbeddingError("Embedding provider request failed; check credentials, quota and connectivity") from None
        return self._validate(vectors, len(texts))

    def embed_query(self, text: str) -> list[float]:
        """短问题使用查询接口；长问题分块编码后均值归一化，不静默丢弃尾部。

        长问题聚合是教学上的折中，不等价于模型一次理解整个问题；多主题查询可能
        稀释相关性。文档仍逐块存储，只有查询会聚合，避免丢失证据粒度。
        """
        chunks = chunk_text(text, overlap=0)
        if not chunks:
            raise EmbeddingError("Embedding query must not be empty")
        if len(chunks) == 1 and self.settings.embedding_provider == "openrouter":
            client = self.client
            try:
                vector = client.embed_query(chunks[0])
            except Exception:
                raise EmbeddingError("Embedding provider request failed; check credentials, quota and connectivity") from None
            return self._validate([vector], 1)[0]
        vectors = self.embed_documents(chunks)
        # 不重叠切分使边界内容不会被重复加权；按字节长度加权后归一化。
        weights = [byte_length(chunk) for chunk in chunks]
        mean = [sum(v[i] * w for v, w in zip(vectors, weights)) / sum(weights) for i in range(self.dim)]
        return self._validate([mean], 1)[0]

    def embed(self, text: str) -> list[float]:
        """兼容旧的单文档入口；新批量写入优先使用 embed_documents。"""
        return self.embed_documents([text])[0]

    def _validate(self, vectors: list[list[float]], count: int) -> list[list[float]]:
        """验证远端返回的每个向量，并做 L2 归一化以稳定余弦检索。"""
        try:
            if len(vectors) != count:
                raise ValueError
            result = []
            for vector in vectors:
                if len(vector) != self.dim or any(isinstance(v, bool) for v in vector):
                    raise ValueError
                values = [float(v) for v in vector]
                norm = math.sqrt(sum(v * v for v in values))
                if not math.isfinite(norm) or norm == 0:
                    raise ValueError
                result.append([v / norm for v in values])
            return result
        except (TypeError, ValueError, OverflowError):
            raise EmbeddingError("Embedding response has an invalid count, dimension or numeric value") from None

    def _hash_embed(self, text: str) -> list[float]:
        """旧演示算法：哈希词频不理解同义词或跨语言语义，保留用于离线对照。"""
        buckets = [0.0] * self.dim
        # 简单按空格切词并去除两端部分标点，不是真正的中文分词器或 LLM tokenizer。
        tokens = [t.strip(".,;:!?()[]{}").lower() for t in text.split()]
        for token in tokens:
            if not token:
                continue
            # SHA-256 在不同进程间稳定；相同词落入相同桶，不同词也可能发生哈希碰撞。
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            # 用另一个哈希字节决定正负贡献，形成有符号的词频投影。
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            buckets[idx] += sign

        # 除以向量长度，使非零向量长度为 1，主要比较方向而非文本长短。
        # 空文本或贡献完全抵消时返回零向量；此兜底仅防除零，不保证余弦检索有效。
        norm = math.sqrt(sum(v * v for v in buckets)) or 1.0
        return [v / norm for v in buckets]
