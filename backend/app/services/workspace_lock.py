"""工作区维护锁：普通请求共享，清空操作独占，避免清空后旧 Agent 又写回数据。

这是诊所级维护互斥，不改变患者共享或会话所有权。连接级锁覆盖业务 commit，
与事务级记录锁不同；所有退出路径都会解锁并归还连接。
"""
from contextlib import contextmanager

from fastapi import Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.db.session import get_db

WORKSPACE_LOCK = 741924


@contextmanager
def workspace_lock(engine: Engine, *, exclusive: bool = False):
    """按当前 schema 加锁，忙时立即返回 409，不终止正在执行的写入。"""
    suffix = "" if exclusive else "_shared"
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        params = {"key": WORKSPACE_LOCK}
        acquired = connection.scalar(text(
            f"SELECT pg_try_advisory_lock{suffix}(:key, hashtext(current_schema()))"), params)
        if not acquired:
            raise HTTPException(409, "Workspace is busy; wait for active requests or maintenance to finish")
        try:
            yield
        finally:
            connection.execute(text(
                f"SELECT pg_advisory_unlock{suffix}(:key, hashtext(current_schema()))"), params)


def workspace_available(db: Session = Depends(get_db)):
    """普通业务 API 的 yield dependency；SSE 生成器还需要自行持锁覆盖整段执行。"""
    with workspace_lock(db.get_bind()):
        yield
