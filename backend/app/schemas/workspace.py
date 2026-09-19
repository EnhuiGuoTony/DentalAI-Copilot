"""前端数据库清空协议；明确确认的是整个共享诊所业务数据，不是某个患者。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict


class WorkspaceResetRequest(BaseModel):
    """只接受固定操作确认，不允许客户端提供表名、SQL 或用户身份。"""
    model_config = ConfigDict(extra="forbid")
    confirmation: Literal["CLEAR_CLINIC_DATA"]


class WorkspaceResetResponse(BaseModel):
    """返回删除数量和当前向量维度；不包含删除前的患者原文或账号信息。"""
    status: Literal["cleared"] = "cleared"
    deleted_counts: dict[str, int]
    embedding_dim: int
