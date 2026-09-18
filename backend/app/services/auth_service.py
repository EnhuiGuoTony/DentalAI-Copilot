"""Cookie 认证边界：随机令牌可撤销，HttpOnly 阻止脚本读取，Origin 检查阻止跨站写入。

这里是单诊所共享工作区，没有把注册用户误当成独立租户；Agent 会话仍需单独检查归属。
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, Request, Response
from pwdlib import PasswordHash
from sqlalchemy.orm import Session
from app.core.config import get_settings
from app.db.models import LoginSession, User
from app.db.session import get_db

password_hash = PasswordHash.recommended()
DUMMY_HASH = password_hash.hash("not-a-real-user-password")
COOKIE = "dentalai_session"


def check_origin(request: Request) -> None:
    """浏览器写操作必须来自配置的前端；无 Origin 的 CLI 调用仍必须通过 Cookie 鉴权。"""
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "Cross-site browser writes are not allowed")
        origin = request.headers.get("origin")
        if origin and origin not in get_settings().allowed_origins:
            raise HTTPException(403, "Untrusted request origin")


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    check_origin(request)
    token = request.cookies.get(COOKIE, "")
    session = db.get(LoginSession, hashlib.sha256(token.encode()).hexdigest()) if token else None
    if session is None or session.expires_at <= datetime.now(timezone.utc):
        raise HTTPException(401, "Please sign in")
    user = db.get(User, session.user_id)
    if user is None:
        raise HTTPException(401, "Please sign in")
    return user


def issue_session(db: Session, user: User, response: Response) -> None:
    """在数据库提交后设置 Cookie；令牌只发给浏览器，不返回密码或哈希。"""
    token = secrets.token_urlsafe(48)
    db.add(LoginSession(token_hash=hashlib.sha256(token.encode()).hexdigest(), user_id=user.id,
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=12)))
    db.commit()
    response.set_cookie(COOKIE, token, httponly=True, secure=get_settings().cookie_secure,
                        samesite="strict", max_age=43200, path="/")
