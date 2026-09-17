import hashlib
import math

from app.core.config import get_settings


class EmbeddingService:
    """无需模型或 API 密钥的确定性演示向量：用词的哈希统计代替语义编码。

    相同文本始终得到相同向量，便于学习入库与检索流程；它没有训练过程，
    不理解同义词、语序或跨语言语义，不能把检索效果当作真实 Embedding 模型的效果。
    聊天模型与向量生成相互独立：配置真实 LLM 不会自动替换这里的哈希算法。
    """

    def __init__(self) -> None:
        # 文档向量、查询向量和数据库 Vector 列必须使用相同维度与编码算法。
        # 更换算法或维度后需要重新生成历史向量；只修改配置不能完成数据迁移。
        self.dim = get_settings().embedding_dim

    def embed(self, text: str) -> list[float]:
        """将文本投影到固定维度并做 L2 归一化，返回可存入 pgvector 的浮点列表。"""
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
