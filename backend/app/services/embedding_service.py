import hashlib
import math

from app.core.config import get_settings


class EmbeddingService:
    """Deterministic local embeddings for a no-key demo.

    This keeps the RAG pipeline runnable before an external embedding provider is configured.
    The vector is not semantically strong, but it exercises chunking, metadata filters,
    vector storage, and retrieval flow.
    """

    def __init__(self) -> None:
        self.dim = get_settings().embedding_dim

    def embed(self, text: str) -> list[float]:
        buckets = [0.0] * self.dim
        tokens = [t.strip(".,;:!?()[]{}").lower() for t in text.split()]
        for token in tokens:
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            buckets[idx] += sign

        norm = math.sqrt(sum(v * v for v in buckets)) or 1.0
        return [v / norm for v in buckets]

