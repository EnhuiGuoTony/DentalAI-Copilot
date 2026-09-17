def chunk_text(text: str, max_words: int = 90, overlap: int = 15) -> list[str]:
    """按空白分词并使用重叠窗口分块，为每块单独生成向量做准备。

    默认每块最多 90 个词，相邻块保留 15 个词，减少信息恰好被边界切断的问题。
    这里的词数不是模型的 token 数；中文连续文本没有空格时可能只算一个词。
    本函数不做参数校验，调用者应保证 max_words > 0 且 0 <= overlap < max_words，
    否则窗口可能无法前进。它也不按段落或句子边界切分，是教学用的简单实现。
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        # 从上一块末尾向前退 overlap 个词：例如第一块 [0:90]，第二块从 75 开始。
        start = max(0, end - overlap)
    return chunks
