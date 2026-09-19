"""清空共享诊所业务数据并重新准备空向量索引；账号、会话凭据与迁移版本保留。

此维护操作仅供用户直接点击，未注册为 Agent 工具，不接受模型生成的操作参数。
它也不连接 DOX 源数据库、不删除上传目录或任何其他数据库/schema。
"""
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.schemas.workspace import WorkspaceResetResponse
from app.services.embedding_index import INDEX_LOCK
from app.services.workspace_lock import workspace_lock

# 明确列出范围，禁止 CASCADE 扩散到未知表；用户账号与登录会话不在清理名单。
BUSINESS_TABLES = (
    "mutation_receipts", "approval_audits", "conversations", "agent_runs",
    "appointments", "embedding_chunks", "knowledge_chunks", "clinical_facts",
    "clinical_notes", "clinical_cases", "dox_import_runs", "patients",
)
CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


def clear_workspace(db: Session) -> WorkspaceResetResponse:
    """在独占维护锁下，一个事务清空业务和 checkpoint，并迁移空表维度。

    正在运行的业务请求/Agent 或索引重建会使操作返回 409；不会中途取消它们。
    SQL 错误会整体回滚，不能出现业务删除但 checkpoint 残留的部分成功。
    """
    with workspace_lock(db.get_bind(), exclusive=True):
        try:
            if not db.scalar(text("SELECT pg_try_advisory_xact_lock(:key, hashtext(current_schema()))"),
                             {"key": INDEX_LOCK}):
                raise HTTPException(409, "Embedding index is busy; wait for the rebuild to finish")
            db.execute(text("SET LOCAL lock_timeout = '5s'"))
            # 精确查询当前 schema，不能误清理 search_path 后面的 public 测试/应用表。
            schema = db.scalar(text("SELECT current_schema()"))
            existing = set(db.scalars(text("SELECT tablename FROM pg_tables WHERE schemaname=:schema"),
                                     {"schema": schema}))
            tables = [*BUSINESS_TABLES, *(name for name in CHECKPOINT_TABLES if name in existing)]
            quote = db.get_bind().dialect.identifier_preparer.quote
            qualified = {name: f"{quote(schema)}.{quote(name)}" for name in tables}
            # 表锁也阻止不认识新维护锁的旧进程与脚本同时改动这些表。
            db.execute(text("LOCK TABLE " + ", ".join(qualified.values()) + " IN ACCESS EXCLUSIVE MODE"))
            counts = {name: db.scalar(text(f"SELECT count(*) FROM {table}")) or 0
                      for name, table in qualified.items()}
            db.execute(text("TRUNCATE TABLE " + ", ".join(qualified.values())))
            dim = get_settings().embedding_dim
            for name in ("embedding_chunks", "knowledge_chunks"):
                db.execute(text(f"ALTER TABLE {qualified[name]} ALTER COLUMN embedding TYPE vector({dim})"))
            db.commit()
            db.info.pop("embedding_index_checked", None)
            return WorkspaceResetResponse(deleted_counts=counts, embedding_dim=dim)
        except HTTPException:
            db.rollback()
            raise
        except SQLAlchemyError as exc:
            db.rollback()
            # 不返回 SQL 参数或数据库异常正文，其中可能包含业务数据。
            if getattr(exc.orig, "sqlstate", None) == "55P03":
                raise HTTPException(409, "Workspace is busy; wait for active writes to finish") from None
            raise HTTPException(503, "Workspace reset failed; no reset changes were committed") from None
