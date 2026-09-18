"""PostgreSQL checkpoint 提供 LangGraph 短期记忆，支持重启后恢复中断。

连接按请求创建并关闭。业务与 checkpoint 是两个事务，写工具用回执防止重复副作用。
"""
from contextlib import contextmanager
from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy.engine import make_url
from app.core.config import get_settings


def checkpoint_url() -> str:
    return make_url(get_settings().database_url).set(drivername="postgresql").render_as_string(hide_password=False)


@contextmanager
def memory():
    with PostgresSaver.from_conn_string(checkpoint_url()) as saver:
        yield saver


def setup_memory() -> None:
    """仅启动时初始化 checkpoint 表，不在每次模型请求里重复迁移。"""
    with memory() as saver:
        saver.setup()
