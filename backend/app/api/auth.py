"""注册、登录、个人资料和改密；改密撤销所有设备会话，要求重新登录。"""
import hashlib
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.db.models import LoginSession, User
from app.db.session import get_db
from app.schemas.auth import Credentials, RegisterRequest, ProfileUpdate, PasswordChange, UserRead
from app.services.auth_service import COOKIE, DUMMY_HASH, check_origin, current_user, issue_session, password_hash

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(check_origin)])


@router.post("/register", response_model=UserRead, status_code=201)
def register(req: RegisterRequest, response: Response, db: Session = Depends(get_db)):
    user = User(username=req.username.lower(), display_name=req.display_name.strip(),
                password_hash=password_hash.hash(req.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Username already exists") from None
    issue_session(db, user, response)
    return user


@router.post("/login", response_model=UserRead)
def login(req: Credentials, response: Response, db: Session = Depends(get_db)):
    # 与改密串行化，避免“旧密码验证通过后、撤销旧会话后”才签发新的有效会话。
    user = db.scalar(select(User).where(User.username == req.username.lower()).with_for_update())
    valid = password_hash.verify(req.password, user.password_hash if user else DUMMY_HASH)
    if not user or not valid:
        raise HTTPException(401, "Invalid username or password")
    issue_session(db, user, response)
    return user


@router.get("/me", response_model=UserRead)
def me(user: User = Depends(current_user)):
    return user


@router.patch("/me", response_model=UserRead)
def update(req: ProfileUpdate, user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.display_name = req.display_name.strip()
    db.commit()
    return user


@router.post("/password", status_code=204)
def change_password(req: PasswordChange, response: Response, user: User = Depends(current_user), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.id == user.id).with_for_update().execution_options(populate_existing=True))
    if not password_hash.verify(req.current_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect")
    user.password_hash = password_hash.hash(req.new_password)
    db.execute(delete(LoginSession).where(LoginSession.user_id == user.id))
    db.commit()
    response.delete_cookie(COOKIE, path="/")


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    token = request.cookies.get(COOKIE, "")
    db.execute(delete(LoginSession).where(LoginSession.token_hash == hashlib.sha256(token.encode()).hexdigest()))
    db.commit()
    response.delete_cookie(COOKIE, path="/")
