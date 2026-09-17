import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.session import Base


settings = get_settings()


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    dox_patient_id: Mapped[str | None] = mapped_column(String(80), nullable=True, unique=True, index=True)
    patient_number: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    medical_record_number: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    cases: Mapped[list["ClinicalCase"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    notes: Mapped[list["ClinicalNote"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class ClinicalCase(Base):
    __tablename__ = "clinical_cases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("patients.id"), index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="draft", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    patient: Mapped[Patient] = relationship(back_populates="cases")


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("patients.id"), index=True)
    case_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("clinical_cases.id"), nullable=True, index=True)
    note_type: Mapped[str] = mapped_column(String(80), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    patient: Mapped[Patient] = relationship(back_populates="notes")


class EmbeddingChunk(Base):
    # 患者 RAG 的检索单元：一条原始笔记可以拆成多个块，每块各有一个向量。
    # patient_id 用于限定检索范围，source_id 用于追溯来源，二者职责不同。
    __tablename__ = "embedding_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("patients.id"), index=True)
    source_type: Mapped[str] = mapped_column(String(80), nullable=False)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Python 属性用 meta，数据库列名用 metadata，避免占用 SQLAlchemy 的 metadata 名称。
    # 元数据可记录笔记类型、来源表和分块序号；保存元数据不等于查询会自动按它过滤。
    meta: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    # 固定维度必须与编码器一致；这个 Vector 列声明本身没有创建 HNSW/IVFFlat 索引。
    # 上面的 index=True 是普通字段索引，不能理解成已经配置向量近似最近邻索引。
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeChunk(Base):
    # 共享知识独立存储，不含 patient_id；它会被全局知识检索工具访问。
    # 仍保留原文和来源，向量只是检索表示，无法从向量可靠还原原始证据。
    __tablename__ = "knowledge_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ClinicalFact(Base):
    # 诊断、治疗等结构化事实供 Agent 直接查询；此表没有 embedding 字段。
    # summary 提供可读摘要，data 保存结构化内容，来源字段便于定位导入记录。
    __tablename__ = "clinical_facts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("patients.id"), index=True)
    fact_type: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    source_system: Mapped[str] = mapped_column(String(80), nullable=False, default="dox")
    source_table: Mapped[str] = mapped_column(String(120), nullable=False)
    source_pk: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(300), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DoxImportRun(Base):
    __tablename__ = "dox_import_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    patients_requested: Mapped[int] = mapped_column(default=0)
    patients_imported: Mapped[int] = mapped_column(default=0)
    notes_imported: Mapped[int] = mapped_column(default=0)
    knowledge_chunks_imported: Mapped[int] = mapped_column(default=0)
    clinical_facts_imported: Mapped[int] = mapped_column(default=0)
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    deidentified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AgentRun(Base):
    # 为保存运行结果和轨迹准备的数据模型；当前 PmsAgentService.run 并未写入此表。
    # “定义了存储表”不等于“已实现持久化对话记忆”。
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("patients.id"), index=True)
    case_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("clinical_cases.id"), index=True)
    user_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_trace: Mapped[list] = mapped_column(JSONB, nullable=False)
    final_output: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
