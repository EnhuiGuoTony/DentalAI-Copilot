from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from app.api import cases, chat, dox_import, patients, rag
from app.api import auth, appointments, conversations
from app.services.auth_service import current_user
from app.services.agent_memory import setup_memory
from app.core.config import get_settings
from app.db.init_db import init_db

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时完成业务表和 checkpoint 迁移，再接受业务请求。"""
    init_db()
    setup_memory()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "app": settings.app_name}


# 统一挂载鉴权，新增业务路由也必须进入此列表，不能只依赖前端隐藏入口。
app.include_router(auth.router, prefix="/api")
for router in (patients.router, cases.router, rag.router, chat.router, dox_import.router,
               appointments.router, conversations.router):
    app.include_router(router, prefix="/api", dependencies=[Depends(current_user)])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
