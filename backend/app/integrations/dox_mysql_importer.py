import re
import uuid
import json
from collections.abc import Iterable
from datetime import date, datetime, timezone
from decimal import Decimal
from html.parser import HTMLParser
from typing import Any

from sqlalchemy import create_engine, text, delete
from sqlalchemy.engine import make_url
from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import ClinicalFact, ClinicalNote, DoxImportRun, EmbeddingChunk, KnowledgeChunk, Patient
from app.schemas.dox_import import DoxImportRequest, DoxImportSummary, DoxPreviewResponse
from app.services.chunking import chunk_text
from app.services.embedding_service import EmbeddingService
from app.services.embedding_index import check_index, index_metadata


DOX_NAMESPACE = uuid.UUID("b9a650f0-c2e8-4ac8-b3af-9a913d3d8ed5")


class _HtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return normalize_text(" ".join(self.parts))


def stable_dox_uuid(kind: str, source_pk: Any, suffix: str | int | None = None) -> uuid.UUID:
    raw = f"{kind}:{source_pk}" if suffix is None else f"{kind}:{source_pk}:{suffix}"
    return uuid.uuid5(DOX_NAMESPACE, raw)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text_value = str(value)
    text_value = re.sub(r"\s+", " ", text_value)
    return text_value.strip()


def html_to_text(value: Any) -> str:
    raw = normalize_text(value)
    if not raw:
        return ""
    parser = _HtmlTextExtractor()
    parser.feed(raw)
    extracted = parser.text()
    return extracted or normalize_text(re.sub(r"<[^>]+>", " ", raw))


def deidentify_clinical_text(value: Any) -> str:
    text_value = normalize_text(value)
    if not text_value:
        return ""
    replacements = [
        (r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", re.IGNORECASE),
        (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", 0),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", 0),
        (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]", 0),
        (r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b", "[DATE]", re.IGNORECASE),
        (r"\b(Doctor|Dr|Hygienist|Assistant|Provider|Staff|Dentist):\s*[^:()]{1,80}(?=\s+[A-Z][A-Za-z ]{2,}:|\s+\(\.\.\.\)|$)", r"\1: [REDACTED]", re.IGNORECASE),
        (r"\b(?:[A-Z][a-z]+),\s*(?:[IVX]+,\s*)?[A-Z][a-z]+\b", "[NAME]", 0),
    ]
    redacted = text_value
    for pattern, replacement, flags in replacements:
        redacted = re.sub(pattern, replacement, redacted, flags=flags)
    return normalize_text(redacted)


def template_content_to_text(value: Any, deidentify: bool = False) -> str:
    raw = normalize_text(value)
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        text_value = html_to_text(raw)
        return deidentify_clinical_text(text_value) if deidentify else text_value

    parts: list[str] = []
    readable_keys = {"plaintext", "html", "_text", "default", "label", "text", "title", "description", "name"}

    def collect(node: Any, key_name: str = "") -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                collect(child, str(key))
        elif isinstance(node, list):
            for child in node:
                collect(child, key_name)
        elif isinstance(node, str) and key_name.lower() in readable_keys:
            text_part = html_to_text(node) if "<" in node and ">" in node else normalize_text(node)
            if text_part:
                parts.append(text_part)

    collect(parsed)
    text_value = normalize_text(" ".join(parts))
    return deidentify_clinical_text(text_value) if deidentify else text_value


def deidentified_patient_name(source_patient_id: int) -> str:
    return f"DOX Patient P{source_patient_id:06d}"


class DoxMySqlImporter:
    """
    Imports useful DOX MySQL data into the portfolio Postgres/pgvector store.

    Design rule:
    - Free text becomes vector-searchable evidence.
    - Tooth, perio, condition, treatment, and attachment metadata becomes structured facts for Agent tools.
    - PHI is deidentified by default.
    """

    def __init__(self, target_db: Session, source_url: str | None = None) -> None:
        self.target_db = target_db
        self.settings = get_settings()
        self.source_url = source_url or self.settings.dox_mysql_url
        self.embedding_service = EmbeddingService()
        self.warnings: list[str] = []

    def preview(self) -> DoxPreviewResponse:
        if not self.source_url:
            return DoxPreviewResponse(connected=False, tables={}, warnings=["DOX_MYSQL_URL is not configured."])

        tables = [
            "Patients",
            "People",
            "Notes",
            "TemplateContents",
            "ClinicalDecisions",
            "Diagnoses",
            "MedicalConditions",
            "PersonConditions",
            "Treatments",
            "Teeth",
            "PatientPerioFindings",
            "EducationalResources",
        ]
        counts: dict[str, int | None] = {}
        engine = self._source_engine()
        with engine.connect() as conn:
            for table_name in tables:
                try:
                    counts[table_name] = int(conn.execute(text(f"SELECT COUNT(*) AS count FROM `{table_name}`")).scalar() or 0)
                except SQLAlchemyError as exc:
                    counts[table_name] = None
                    self.warnings.append(f"{table_name}: {self._short_error(exc)}")
        return DoxPreviewResponse(connected=True, tables=counts, warnings=self.warnings)

    def import_data(self, req: DoxImportRequest) -> DoxImportSummary:
        if not self.source_url:
            return DoxImportSummary(status="blocked", warnings=["DOX_MYSQL_URL is not configured."])

        run = DoxImportRun(
            status="running",
            patients_requested=len(req.patient_ids or []),
            deidentified=req.deidentify,
            warnings=[],
        )
        self.target_db.add(run)
        self.target_db.commit()
        self.target_db.refresh(run)

        summary = DoxImportSummary(run_id=str(run.id), status="running", started_at=run.started_at)
        try:
            check_index(self.target_db, self.embedding_service)
            engine = self._source_engine()
            patient_ids = req.patient_ids or self._load_patient_ids(engine, req.patient_limit or self.settings.dox_import_patient_limit)
            summary.patients_requested = len(patient_ids)

            if req.include_global_knowledge:
                summary.knowledge_chunks_imported += self._import_global_knowledge(engine, req.deidentify)

            if req.include_patient_history:
                for source_patient_id in patient_ids:
                    patient = self._upsert_patient(engine, source_patient_id, req.deidentify)
                    summary.patients_imported += 1
                    summary.notes_imported += self._import_patient_notes(engine, source_patient_id, patient.id, req.deidentify)
                    summary.clinical_facts_imported += self._import_patient_facts(engine, source_patient_id, patient.id, req.deidentify)

            summary.status = "completed_with_warnings" if self.warnings else "completed"
        except (SQLAlchemyError, TypeError, ValueError) as exc:
            self.target_db.rollback()
            summary.status = "failed"
            self.warnings.append(self._short_error(exc))
        finally:
            summary.warnings = self.warnings
            summary.finished_at = datetime.now(timezone.utc)
            run.status = summary.status
            run.patients_requested = summary.patients_requested
            run.patients_imported = summary.patients_imported
            run.notes_imported = summary.notes_imported
            run.knowledge_chunks_imported = summary.knowledge_chunks_imported
            run.clinical_facts_imported = summary.clinical_facts_imported
            run.warnings = summary.warnings
            run.finished_at = summary.finished_at
            self.target_db.merge(run)
            self.target_db.commit()

        return summary

    def _source_engine(self) -> Engine:
        source_url = make_url(self.source_url)
        if self.settings.dox_mysql_host:
            source_url = source_url.set(host=self.settings.dox_mysql_host)
        return create_engine(source_url, pool_pre_ping=True)

    def _load_patient_ids(self, engine: Engine, limit: int) -> list[int]:
        safe_limit = max(1, min(limit, 500))
        # DOX relationship tables (Notes, Treatments, PersonConditions, etc.)
        # reference Patients.ID.  PatientID is the practice-visible number and
        # is retained separately in patients.dox_patient_id.
        rows = self._safe_fetch(engine, "Patients", f"SELECT ID FROM `Patients` ORDER BY ID LIMIT {safe_limit}")
        return [int(row["ID"]) for row in rows if row.get("ID") is not None]

    def _upsert_patient(self, engine: Engine, source_patient_id: int, deidentify: bool) -> Patient:
        patient_id = stable_dox_uuid("Patients", source_patient_id)
        rows = self._safe_fetch(
            engine,
            "Patients/People/Locations",
            """
            SELECT p.PatientID, p.PatientNumber, p.FileNumber,
                   pe.FirstName, pe.MiddleName, pe.LastName, pe.DateOfBirth,
                   l.AddressLine1, l.AddressLine2, l.City, l.StateOrTerritory, l.PostalCode
            FROM `Patients` p
            LEFT JOIN `People` pe ON pe.ID = p.ID
            LEFT JOIN `PersonLocations` pl ON pl.PersonID = pe.ID
            LEFT JOIN `Locations` l ON l.ID = pl.LocationID
            WHERE p.ID = :patient_id
            ORDER BY pl.IsDefault DESC, pl.Sequence ASC
            LIMIT 1
            """,
            {"patient_id": source_patient_id},
        )
        source = rows[0] if rows else {}
        name = " ".join(
            part for part in (normalize_text(source.get("FirstName")), normalize_text(source.get("MiddleName")), normalize_text(source.get("LastName"))) if part
        ) or deidentified_patient_name(source_patient_id)
        address = ", ".join(
            part for part in (
                normalize_text(source.get("AddressLine1")), normalize_text(source.get("AddressLine2")),
                normalize_text(source.get("City")), normalize_text(source.get("StateOrTerritory")), normalize_text(source.get("PostalCode")),
            ) if part
        ) or None
        patient = self.target_db.get(Patient, patient_id)
        if patient is None:
            patient = Patient(id=patient_id)
        if deidentify:
            patient.name = deidentified_patient_name(source_patient_id)
            patient.date_of_birth = None
            patient.address = None
        else:
            patient.name = name
            patient.date_of_birth = source.get("DateOfBirth")
            patient.address = address
        patient.dox_patient_id = str(source.get("PatientID") or source_patient_id)
        patient.patient_number = normalize_text(source.get("PatientNumber")) or None
        patient.medical_record_number = normalize_text(source.get("FileNumber")) or None
        self.target_db.merge(patient)
        self.target_db.commit()
        return self.target_db.get(Patient, patient_id) or patient

    def _import_patient_notes(self, engine: Engine, source_patient_id: int, patient_id: uuid.UUID, deidentify: bool) -> int:
        """把源系统笔记同时导入原文表和患者向量表，建立可追溯的检索材料。

        原文用于时间线与最近笔记工具，分块用于 RAG；同一材料服务于不同查询方式。
        HTML 先转纯文本，按请求决定是否脱敏，随后再分块生成向量。
        """
        rows = self._safe_fetch(
            engine,
            "Notes",
            """
            SELECT ID, PatientID, AppointmentID, TypeID, KindOfNote, Name, Description, Content, IsHTML
            FROM `Notes`
            WHERE PatientID = :patient_id AND IsDeleted = 0 AND Content IS NOT NULL AND TRIM(Content) <> ''
            ORDER BY ID DESC
            """,
            {"patient_id": source_patient_id},
        )
        imported = 0
        for row in rows:
            text_value = html_to_text(row.get("Content")) if row.get("IsHTML") else normalize_text(row.get("Content"))
            if deidentify:
                text_value = deidentify_clinical_text(text_value)
            if not text_value:
                continue
            note_id = stable_dox_uuid("Notes", row["ID"])
            note = ClinicalNote(
                id=note_id,
                patient_id=patient_id,
                note_type=f"dox_note:{row.get('KindOfNote') or row.get('TypeID') or 'unknown'}",
                content=text_value,
            )
            self.target_db.merge(note)
            self._upsert_patient_chunks(
                patient_id=patient_id,
                source_type="dox_note",
                source_id=note_id,
                text_value=text_value,
                metadata={
                    "source_system": "dox",
                    "source_table": "Notes",
                    "source_pk": str(row["ID"]),
                    "source_patient_ref": str(stable_dox_uuid("SourcePatientRef", source_patient_id)),
                    "name": deidentify_clinical_text(row.get("Name")) if deidentify else normalize_text(row.get("Name")),
                    "description": deidentify_clinical_text(row.get("Description")) if deidentify else normalize_text(row.get("Description")),
                    "appointment_id": row.get("AppointmentID"),
                    "kind_of_note": row.get("KindOfNote"),
                    "type_id": row.get("TypeID"),
                    "deidentified": deidentify,
                },
            )
            imported += 1
        self.target_db.commit()
        return imported

    def _import_global_knowledge(self, engine: Engine, deidentify: bool) -> int:
        imported = 0
        imported += self._import_knowledge_query(
            engine,
            "ClinicalDecisions",
            """
            SELECT ID, Name, Description, SnomedConceptID, SnomedConceptDescription, ReferenceLink,
                   DocumentationReleaseDate, DocumentationRevisionDate, DecisionType
            FROM `ClinicalDecisions`
            WHERE IsActive = 1
            """,
            "dox_clinical_decision",
            ["Name", "Description", "SnomedConceptDescription", "ReferenceLink"],
            deidentify=deidentify,
        )
        imported += self._import_knowledge_query(
            engine,
            "Diagnoses",
            """
            SELECT ID, Name, Description, ICDAS_Value, ICD10Code, ICD10Name, SnomedCode, SnomedName
            FROM `Diagnoses`
            """,
            "dox_diagnosis",
            ["Name", "Description", "ICD10Code", "ICD10Name", "SnomedCode", "SnomedName"],
            deidentify=deidentify,
        )
        imported += self._import_knowledge_query(
            engine,
            "MedicalConditions",
            """
            SELECT ID, Name, Description, IsAlert, AllergyCategory, AllergyRxNormID, ExternalMappings, ODM_TCode
            FROM `MedicalConditions`
            WHERE IsActive = 1
            """,
            "dox_medical_condition",
            ["Name", "Description", "AllergyRxNormID", "ODM_TCode"],
            deidentify=deidentify,
        )
        imported += self._import_knowledge_query(
            engine,
            "TemplateContents",
            """
            SELECT ID, NoteTemplateID, LetterTemplateID, EmailTemplateID, TextMessageTemplateID,
                   FormTemplateID, Category, Content
            FROM `TemplateContents`
            WHERE IsActive = 1 AND Content IS NOT NULL AND TRIM(Content) <> ''
            """,
            "dox_template_content",
            ["Content"],
            html_fields=["Content"],
            deidentify=deidentify,
        )
        imported += self._import_knowledge_query(
            engine,
            "EducationalResources",
            """
            SELECT ID, Name, Description, URL, SnomedConceptID, SnomedConceptDescription
            FROM `EducationalResources`
            WHERE IsActive = 1
            """,
            "dox_educational_resource",
            ["Name", "Description", "URL", "SnomedConceptDescription"],
            deidentify=deidentify,
        )
        self.target_db.commit()
        return imported

    def _import_knowledge_query(
        self,
        engine: Engine,
        table_name: str,
        sql: str,
        source_type: str,
        text_fields: list[str],
        html_fields: list[str] | None = None,
        deidentify: bool = False,
    ) -> int:
        rows = self._safe_fetch(engine, table_name, sql)
        imported = 0
        html_field_set = set(html_fields or [])
        for row in rows:
            source_pk = row.get("ID")
            parts = []
            for field in text_fields:
                if table_name == "TemplateContents" and field == "Content":
                    value = template_content_to_text(row.get(field), deidentify=deidentify)
                else:
                    value = html_to_text(row.get(field)) if field in html_field_set else normalize_text(row.get(field))
                if deidentify and not (table_name == "TemplateContents" and field == "Content"):
                    value = deidentify_clinical_text(value)
                if value:
                    parts.append(f"{field}: {value}")
            text_value = "\n".join(parts)
            if not text_value or source_pk is None:
                continue
            # 共享知识保留字段名称以提供上下文，再按块编码；它不绑定具体患者。
            # 稳定 UUID 使相同来源与块序号在重复导入时定位到相同记录。
            source_id = stable_dox_uuid(table_name, source_pk)
            check_index(self.target_db, self.embedding_service)
            chunks = chunk_text(text_value)
            vectors = self.embedding_service.embed_documents(chunks)
            # 编码成功后替换该来源的全部旧块，原文缩短也不会留下多余历史片段。
            self.target_db.execute(delete(KnowledgeChunk).where(
                KnowledgeChunk.source_type == source_type, KnowledgeChunk.source_id == source_id))
            for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                chunk_id = stable_dox_uuid(f"KnowledgeChunk:{table_name}", source_pk, idx)
                self.target_db.merge(
                    KnowledgeChunk(
                        id=chunk_id,
                        source_type=source_type,
                        source_id=source_id,
                        chunk_text=chunk,
                        meta=index_metadata(self.embedding_service, {
                            "source_system": "dox",
                            "source_table": table_name,
                            "source_pk": str(source_pk),
                            "chunk_index": idx,
                        }),
                        embedding=vector,
                    )
                )
                imported += 1
        return imported

    def _import_patient_facts(self, engine: Engine, source_patient_id: int, patient_id: uuid.UUID, deidentify: bool) -> int:
        imported = 0
        imported += self._import_fact_query(
            engine,
            "PersonConditions",
            """
            SELECT pc.ID, pc.Description, pc.ToothID, pc.ToothSurfaces, pc.DiagnosisDate, pc.ReportedDate,
                   pc.ResolvedDate, pc.ConfirmationStatus, mc.Name AS MedicalConditionName,
                   d.Name AS DiagnosisName, d.ICD10Code, d.SnomedCode
            FROM `PersonConditions` pc
            LEFT JOIN `MedicalConditions` mc ON mc.ID = pc.MedicalConditionID
            LEFT JOIN `Diagnoses` d ON d.ID = pc.DiagnosisID
            WHERE pc.PersonID = :patient_id AND pc.IsDeleted = 0
            """,
            {"patient_id": source_patient_id},
            patient_id,
            "condition",
            "Condition",
            ["MedicalConditionName", "DiagnosisName", "Description", "ICD10Code", "SnomedCode"],
            "DiagnosisDate", deidentify,
        )
        imported += self._import_fact_query(
            engine,
            "Treatments",
            """
            SELECT t.ID, t.ToothID, t.ToothLocation, t.ToothType, t.Surfaces, t.TreatmentDate, t.Status,
                   t.Minutes, t.Fee, p.ProcedureCode, p.Name AS ProcedureName, p.Description AS ProcedureDescription
            FROM `Treatments` t
            LEFT JOIN `Procedures` p ON p.ID = t.ProcedureID
            WHERE t.PatientID = :patient_id AND t.IsDeleted = 0
            """,
            {"patient_id": source_patient_id},
            patient_id,
            "treatment",
            "Treatment",
            ["ProcedureCode", "ProcedureName", "ProcedureDescription", "ToothID", "Surfaces", "Status"],
            "TreatmentDate", deidentify,
        )
        imported += self._import_fact_query(
            engine,
            "PatientPerioFindings",
            """
            SELECT ID, PatientExamID, Location, BleedingSites, PlaqueSites, CalculusSites,
                   InfectionSites, Pocket_DB, Pocket_B, Pocket_MB, Pocket_DL, Pocket_L, Pocket_ML,
                   Recession_DB, Recession_B, Recession_MB, Recession_DL, Recession_L, Recession_ML
            FROM `PatientPerioFindings`
            WHERE PatientID = :patient_id
            """,
            {"patient_id": source_patient_id},
            patient_id,
            "perio_finding",
            "Perio",
            ["Location", "BleedingSites", "PlaqueSites", "CalculusSites", "InfectionSites", "Pocket_DB", "Pocket_B", "Pocket_MB", "Pocket_DL", "Pocket_L", "Pocket_ML"],
            None, deidentify,
        )
        self.target_db.commit()
        return imported

    def _import_fact_query(
        self,
        engine: Engine,
        table_name: str,
        sql: str,
        params: dict[str, Any],
        patient_id: uuid.UUID,
        fact_type: str,
        label_prefix: str,
        summary_fields: list[str],
        effective_field: str | None,
        deidentify: bool,
    ) -> int:
        rows = self._safe_fetch(engine, table_name, sql, params)
        imported = 0
        for row in rows:
            source_pk = row.get("ID")
            if source_pk is None:
                continue
            summary_parts = [normalize_text(row.get(field)) for field in summary_fields]
            summary = "; ".join(part for part in summary_parts if part)
            if deidentify:
                summary = deidentify_clinical_text(summary)
            if not summary:
                summary = f"{label_prefix} imported from DOX {table_name} #{source_pk}"
            # 结构化事实直接保存业务字段和摘要，不在此处向量化。
            # Agent 可用 SQL 工具按患者读取这些记录，不必所有问题都走 RAG。
            fact = ClinicalFact(
                id=stable_dox_uuid(f"ClinicalFact:{table_name}", source_pk),
                patient_id=patient_id,
                fact_type=fact_type,
                source_system="dox",
                source_table=table_name,
                source_pk=str(source_pk),
                label=f"{label_prefix} #{source_pk}",
                summary=summary,
                data=self._deidentify_data(self._jsonable(row)) if deidentify else self._jsonable(row),
                effective_at=row.get(effective_field) if effective_field else None,
            )
            self.target_db.merge(fact)
            imported += 1
        return imported

    def _upsert_patient_chunks(
        self,
        patient_id: uuid.UUID,
        source_type: str,
        source_id: uuid.UUID,
        text_value: str,
        metadata: dict[str, Any],
    ) -> None:
        """按来源和分块序号更新患者向量块，保留患者范围与来源元数据。

        先完成批量编码，再删除同患者、同来源的旧块并写入新块，避免缩短后残留。
        删除和插入仍在调用者的事务里；失败回滚后旧索引保持不变。
        """
        check_index(self.target_db, self.embedding_service)
        chunks = chunk_text(text_value)
        vectors = self.embedding_service.embed_documents(chunks)
        self.target_db.execute(delete(EmbeddingChunk).where(
            EmbeddingChunk.patient_id == patient_id, EmbeddingChunk.source_type == source_type,
            EmbeddingChunk.source_id == source_id))
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            self.target_db.merge(
                EmbeddingChunk(
                    id=stable_dox_uuid(f"EmbeddingChunk:{source_type}", source_id, idx),
                    patient_id=patient_id,
                    source_type=source_type,
                    source_id=source_id,
                    chunk_text=chunk,
                    meta=index_metadata(self.embedding_service, {**metadata, "chunk_index": idx}),
                    embedding=vector,
                )
            )

    @staticmethod
    def _deidentify_data(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: DoxMySqlImporter._deidentify_data(item) for key, item in value.items()}
        if isinstance(value, list):
            return [DoxMySqlImporter._deidentify_data(item) for item in value]
        return deidentify_clinical_text(value) if isinstance(value, str) else value

    def _safe_fetch(
        self,
        engine: Engine,
        table_name: str,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> list[RowMapping]:
        try:
            with engine.connect() as conn:
                return list(conn.execute(text(sql), params or {}).mappings().all())
        except SQLAlchemyError as exc:
            self.warnings.append(f"{table_name}: {self._short_error(exc)}")
            return []

    @staticmethod
    def _jsonable(row: RowMapping | Iterable[tuple[str, Any]]) -> dict[str, Any]:
        source = dict(row)
        result: dict[str, Any] = {}
        for key, value in source.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, date):
                result[key] = value.isoformat()
            elif isinstance(value, Decimal):
                result[key] = float(value)
            elif isinstance(value, bytes):
                result[key] = value.hex()
            else:
                result[key] = value
        return result

    @staticmethod
    def _short_error(exc: Exception) -> str:
        return normalize_text(str(exc)).split("(Background on this error")[0][:500]
