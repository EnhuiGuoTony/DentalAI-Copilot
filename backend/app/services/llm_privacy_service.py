"""在模型边界前替换已知姓名、出生日期和地址，不修改数据库中的原始记录。

当前 Agent 在返回患者工具结果时调用本服务；不要据此假定所有发给模型的文本
都已经脱敏。这里使用字段规则和已知值匹配，不是完整的敏感信息识别系统，
也不自动覆盖患者编号、电话等其他标识。
"""

from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Patient


class LlmPrivacyService:
    _field_tokens = {
        "name": "[PATIENT_NAME]",
        "patient_name": "[PATIENT_NAME]",
        "date_of_birth": "[DATE_OF_BIRTH]",
        "dob": "[DATE_OF_BIRTH]",
        "address": "[ADDRESS]",
    }

    def __init__(self, db: Session) -> None:
        # 读取全库已知值作为替换字典，以处理自由文本中出现的姓名、地址和生日。
        # 长文本先替换，避免短姓名/地址片段先命中后破坏较长条目的匹配。
        self.replacements: list[tuple[str, str]] = []
        for patient in db.execute(select(Patient)).scalars():
            self._add(patient.name, "[PATIENT_NAME]")
            self._add_date(patient.date_of_birth)
            self._add(patient.address, "[ADDRESS]")
        self.replacements.sort(key=lambda item: len(item[0]), reverse=True)

    def redact(self, value: Any, field_name: str = "") -> Any:
        """递归创建脱敏后的字典/列表；非字符串数据尽量保留类型与业务含义。"""
        # 先按字段名处理，再按具体值匹配：即使生日格式不同，也可通过字段规则识别。
        field_token = self._field_tokens.get(field_name.lower())
        if field_token and value not in (None, ""):
            return field_token
        if isinstance(value, dict):
            return {key: self.redact(item, key) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, (date, datetime)):
            return "[DATE_OF_BIRTH]" if field_name.lower() in {"date_of_birth", "dob"} else value.isoformat()
        if not isinstance(value, str):
            return value
        result = value
        for source, replacement in self.replacements:
            # escape 将源文本视为字面量，避免姓名中的字符被解释为正则语法。
            # 忽略大小写但仍是子串替换，可能漏掉变体，也可能误替换普通文本。
            result = re.sub(re.escape(source), replacement, result, flags=re.IGNORECASE)
        return result

    def _add(self, value: str | None, replacement: str) -> None:
        normalized = (value or "").strip()
        if normalized:
            self.replacements.append((normalized, replacement))

    def _add_date(self, value: date | None) -> None:
        """补充 ISO 和两种月/日/年格式；这不是对所有日期表达形式的通用识别。"""
        if value is None:
            return
        self._add(value.isoformat(), "[DATE_OF_BIRTH]")
        self._add(f"{value.month}/{value.day}/{value.year}", "[DATE_OF_BIRTH]")
        self._add(f"{value.month:02d}/{value.day:02d}/{value.year}", "[DATE_OF_BIRTH]")
