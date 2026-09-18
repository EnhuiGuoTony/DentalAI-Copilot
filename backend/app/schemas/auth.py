"""账号 API 契约；输入长度限制也约束密码哈希的资源消耗。"""
from datetime import datetime
from uuid import UUID
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class Credentials(BaseModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_.-]{3,80}$")
    password: str = Field(min_length=10, max_length=128)


class RegisterRequest(Credentials):
    # 只清理显示名称，不得静默修改用户密码中的空格。
    display_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    display_name: str = Field(min_length=1, max_length=120)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=10, max_length=128)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    username: str
    display_name: str
    created_at: datetime
