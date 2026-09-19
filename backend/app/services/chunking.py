"""RAG 共用切片入口：导入、笔记写入和索引重建使用相同的 LangChain 策略。"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import get_settings


def byte_length(text: str) -> int:
    """按 UTF-8 字节计量，中文和 emoji 比 ASCII 占用更多预算；不是 token 计数器。"""
    return len(text.encode("utf-8"))


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """先按段落、句子和词边界递归切片，超长无分隔文本最终按字符切分。

    默认每块最多 480 UTF-8 字节、目标重叠 72 字节。实际重叠随自然边界变化。
    原来的 max_words 已改为字节预算；该保守策略不需要下载模型 tokenizer。
    """
    settings = get_settings()
    size = settings.rag_chunk_size if chunk_size is None else chunk_size
    shared = settings.rag_chunk_overlap if overlap is None else overlap
    if not 4 <= size <= 480 or not 0 <= shared < size:
        raise ValueError("Chunk size must be 4..480 bytes and overlap must be smaller than size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=shared, length_function=byte_length,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", "，", ", ", " ", ""],
        keep_separator="end",
    )
    return splitter.split_text(text) if text.strip() else []
