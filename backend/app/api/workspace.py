"""诊所维护入口，由 main.py 挂载 current_user，沿用 Cookie 与 Origin 校验。"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.workspace import WorkspaceResetRequest, WorkspaceResetResponse
from app.services.workspace_service import clear_workspace

router = APIRouter(prefix="/workspace", tags=["workspace"])


@router.post("/reset", response_model=WorkspaceResetResponse)
def reset_workspace(req: WorkspaceResetRequest, db: Session = Depends(get_db)):
    """确认字段通过 Pydantic 校验后清空共享业务数据；不是模型工具或审批恢复入口。"""
    return clear_workspace(db)
