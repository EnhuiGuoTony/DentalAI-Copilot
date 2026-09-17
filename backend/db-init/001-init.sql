-- 在 PostgreSQL 中启用 pgvector 的向量类型与距离运算；这里不创建向量检索索引。
-- 扩展用于存储和比较向量，不负责生成向量，向量由应用层 EmbeddingService 提供。
CREATE EXTENSION IF NOT EXISTS vector;
