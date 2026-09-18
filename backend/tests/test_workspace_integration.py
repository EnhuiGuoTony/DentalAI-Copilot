"""真实 PostgreSQL 集成测试：独立随机 schema，不读取或修改原有患者数据。

运行前设置 TEST_DATABASE_URL 为具有 CREATE SCHEMA 权限、已安装 vector 扩展的测试库。
未设置时明确跳过；可控模型验证真实 LangChain HITL 与 PostgresSaver，不调用外部 LLM。
"""
import json
import os
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from sqlalchemy import create_engine, select, func, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from app.core.config import get_settings
from app.db.session import Base, get_db
from app.db.models import Patient, ClinicalNote, EmbeddingChunk, Appointment, MutationReceipt
from app.main import app
from app.services.agent_memory import setup_memory
from app.services import pms_agent_service, conversation_service
from app.services.records_service import apply_change, patient_version, version
from app.schemas.operations import ChangeRequest


@pytest.fixture
def workspace(monkeypatch):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL integration tests")
    schema = "test_workspace_" + uuid4().hex
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    test_url = make_url(url).update_query_dict({"options": f"-csearch_path={schema},public"})
    engine = create_engine(test_url)
    factory = sessionmaker(engine, autoflush=False)
    try:
        # checkfirst=False 避免把 public 中同名业务表误认为测试表已经存在。
        Base.metadata.create_all(engine, checkfirst=False)
        # checkpoint 建表时不搜索 public，避免误复用正在运行应用的同名 checkpoint 表。
        checkpoint_test_url = test_url.update_query_dict({"options": f"-csearch_path={schema}"})
        monkeypatch.setattr(get_settings(), "database_url", checkpoint_test_url.render_as_string(hide_password=False))
        monkeypatch.setattr(get_settings(), "mock_llm", True)
        monkeypatch.setattr(pms_agent_service, "SessionLocal", factory)
        monkeypatch.setattr(conversation_service, "SessionLocal", factory)
        monkeypatch.setattr(conversation_service, "engine", engine)
        setup_memory()

        def db_override():
            with factory() as db:
                yield db
        app.dependency_overrides[get_db] = db_override
        yield factory
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
        # 名称由本测试固定前缀和 UUID 构造，只清理本次创建的 schema。
        assert schema.startswith("test_workspace_") and len(schema) == 47
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def register(client, username="doctor_one"):
    response = client.post("/api/auth/register", json={"username": username, "password": "Test-password-123", "display_name": "Demo clinician"})
    assert response.status_code == 201, response.text
    return response.json()


def events(response):
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_all_business_routes_require_login(workspace):
    """即使请求体不完整，未认证的业务请求仍先被服务端拒绝。"""
    client = TestClient(app)
    for route in app.routes:
        if route.path.startswith("/api/") and not route.path.startswith("/api/auth/"):
            path = route.path
            for name in ("patient_id", "case_id", "note_id", "appointment_id", "conversation_id"):
                path = path.replace("{" + name + "}", str(uuid4()))
            for method in route.methods:
                response = client.request(method, path, json={})
                assert response.status_code == 401, (method, path, response.text)


def test_accounts_revocation_and_origin(workspace):
    client = TestClient(app)
    user = register(client)
    assert "password_hash" not in user
    assert "HttpOnly" in client.post("/api/auth/login", json={"username": "doctor_one", "password": "Test-password-123"}).headers["set-cookie"]
    other_device = TestClient(app)
    assert other_device.post("/api/auth/login", json={"username": "DOCTOR_ONE", "password": "Test-password-123"}).status_code == 200
    assert client.patch("/api/auth/me", json={"display_name": "Changed"}).json()["display_name"] == "Changed"
    assert client.post("/api/patients", headers={"Origin": "https://untrusted.example"}, json={"name": "Blocked"}).status_code == 403
    assert client.post("/api/auth/password", json={"current_password": "wrong", "new_password": "New-password-123"}).status_code == 400
    assert client.post("/api/auth/password", json={"current_password": "Test-password-123", "new_password": "New-password-123"}).status_code == 204
    assert other_device.get("/api/patients").status_code == 401
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/login", json={"username": "doctor_one", "password": "New-password-123"}).status_code == 200
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/patients").status_code == 401


def test_appointment_conflicts_and_versions(workspace):
    client = TestClient(app); register(client)
    patient = client.post("/api/patients", json={"name": "Synthetic patient"}).json()
    start = datetime.now(timezone.utc) + timedelta(days=1)
    data = {"patient_id": patient["id"], "starts_at": start.isoformat(), "ends_at": (start + timedelta(minutes=30)).isoformat(), "reason": "Demo visit"}
    assert client.post("/api/appointments", json=data).status_code == 201
    assert client.post("/api/appointments", json=data).status_code == 409
    assert client.post("/api/appointments", json={**data, "ends_at": data["starts_at"]}).status_code == 422
    row = client.get("/api/appointments").json()[0]
    update = {"operation": "update_appointment", "patient_id": patient["id"], "appointment_id": row["id"], "expected_version": row["version"], "data": {k: v for k, v in data.items() if k != "patient_id"}}
    update["data"]["reason"] = "Updated visit"
    assert client.put(f'/api/appointments/{row["id"]}', json=update).status_code == 200
    assert client.put(f'/api/appointments/{row["id"]}', json=update).status_code == 409
    row = client.get("/api/appointments").json()[0]
    assert client.delete(f'/api/appointments/{row["id"]}', params={"patient_id": patient["id"], "expected_version": row["version"]}).status_code == 200
    assert client.get("/api/appointments").json() == []


def test_note_indexing_and_patient_delete_cleanup(workspace):
    client = TestClient(app); register(client)
    patient = client.post("/api/patients", json={"name": "Synthetic patient"}).json()
    note = client.post(f'/api/patients/{patient["id"]}/notes', json={"content": "Synthetic dental history for testing."}).json()
    with workspace() as db:
        assert db.scalar(select(func.count()).select_from(EmbeddingChunk)) > 0
        n = db.get(ClinicalNote, UUID(note["id"]))
        note_version = version(n)
    for _ in range(2):
        assert client.post(f'/api/patients/{patient["id"]}/notes/ingest').status_code == 200
    with workspace() as db:
        assert db.scalar(select(func.count()).select_from(EmbeddingChunk)) == 1
    assert client.delete(f'/api/patients/{patient["id"]}/notes/{note["id"]}', params={"expected_version": note_version}).status_code == 200
    client.post(f'/api/patients/{patient["id"]}/notes', json={"content": "Another synthetic note."})
    v = client.get(f'/api/patients/{patient["id"]}/version').json()["version"]
    assert client.delete(f'/api/patients/{patient["id"]}', params={"expected_version": v}).status_code == 200
    with workspace() as db:
        for model in (Patient, ClinicalNote, EmbeddingChunk):
            assert db.scalar(select(func.count()).select_from(model)) == 0


class WriteModel(BaseChatModel):
    """模拟 provider 的工具协议，但审批和数据库执行完全使用真实业务代码。"""
    planned_change: dict | None = None
    @property
    def _llm_type(self): return "test-tool-model"
    def bind_tools(self, tools, **kwargs): return self
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        completed = any(isinstance(m, ToolMessage) and m.name == "change_records" for m in messages)
        call = ({"name": "AgentAnswer", "id": "answer", "args": {"answer": "Finished", "evidence_ids": [], "limitations": []}}
                if completed else {"name": "change_records", "id": "write-once", "args": {"change": self.planned_change or {"operation": "create_patient", "data": {"name": "Agent synthetic patient"}}}})
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


@pytest.fixture
def write_model(monkeypatch):
    original = pms_agent_service.PmsAgentService.build
    monkeypatch.setattr(pms_agent_service.PmsAgentService, "build", lambda self, saver, model=None: original(self, saver, WriteModel()))


def test_hitl_persists_isolates_and_executes_once(workspace, write_model):
    owner = TestClient(app); user = register(owner)
    stranger = TestClient(app); register(stranger, "doctor_two")
    cid = owner.post("/api/conversations", json={"patient_ids": []}).json()["id"]
    run = events(owner.post(f"/api/conversations/{cid}/messages", json={"message": "Create a synthetic patient"}))
    assert run[-1]["type"] == "approval", run
    with workspace() as db:
        assert db.scalar(select(func.count()).select_from(Patient)) == 0
    # 新建图与 PostgresSaver 连接读取旧 checkpoint，模拟进程重启后的恢复。
    restored = owner.get(f"/api/conversations/{cid}").json()
    review = restored["review"]
    assert review["interrupt_id"] == run[-1]["review"]["interrupt_id"]
    request = {"interrupt_id": review["interrupt_id"], "decisions": [{"type": "approve"}]}
    assert stranger.get(f"/api/conversations/{cid}").status_code == 404
    assert stranger.post(f"/api/conversations/{cid}/resume", json=request).status_code == 404
    assert stranger.get("/api/conversations").json() == []
    assert events(owner.post(f"/api/conversations/{cid}/resume", json={**request, "interrupt_id": "stale"}))[-1]["type"] == "error"
    assert events(owner.post(f"/api/conversations/{cid}/messages", json={"message": "Skip approval"}))[-1]["type"] == "error"
    # 保存暂停 checkpoint，随后用它重放写节点，验证 DB 回执与 checkpoint 间的崩溃窗口。
    with conversation_service.graph_context(UUID(cid), UUID(user["id"])) as (graph, config, service):
        paused = graph.get_state(config).config
    result = events(owner.post(f"/api/conversations/{cid}/resume", json=request))
    assert result[-1]["type"] == "result", result
    assert events(owner.post(f"/api/conversations/{cid}/resume", json=request))[-1]["type"] == "error"
    from langgraph.types import Command
    with conversation_service.graph_context(UUID(cid), UUID(user["id"])) as (graph, config, service):
        graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=paused)
    with workspace() as db:
        assert db.scalar(select(func.count()).select_from(Patient)) == 1
        assert db.scalar(select(func.count()).select_from(MutationReceipt)) == 1


def test_hitl_reject_does_not_write(workspace, write_model):
    client = TestClient(app); register(client)
    cid = client.post("/api/conversations", json={}).json()["id"]
    review = events(client.post(f"/api/conversations/{cid}/messages", json={"message": "Create a patient"}))[-1]["review"]
    result = events(client.post(f"/api/conversations/{cid}/resume", json={"interrupt_id": review["interrupt_id"], "decisions": [{"type": "reject"}]}))
    assert result[-1]["type"] == "result", result
    with workspace() as db:
        assert db.scalar(select(func.count()).select_from(Patient)) == 0


def test_mock_memory_and_fixed_patient_scope(workspace):
    client = TestClient(app); register(client)
    cid = client.post("/api/conversations", json={}).json()["id"]
    for message in ("First message", "Second message"):
        result = events(client.post(f"/api/conversations/{cid}/messages", json={"message": message}))
        assert result[-1]["type"] == "result" and result[-1]["mock"], result
    state = client.get(f"/api/conversations/{cid}").json()
    assert [m["content"] for m in state["messages"] if m["role"] == "user"] == ["First message", "Second message"]
    assert len([m for m in state["messages"] if m["role"] == "assistant"]) == 2
    with workspace() as db:
        patient = Patient(name="Out of scope")
        db.add(patient); db.commit()
        change = ChangeRequest.model_validate({"change": {"operation": "add_note", "patient_id": str(patient.id), "content": "Blocked"}}).change
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as error:
            apply_change(db, change, allowed_ids=[])
        assert error.value.status_code == 403


@pytest.mark.parametrize("operation", ["update_patient", "delete_patient", "create_appointment", "update_appointment", "delete_appointment", "add_note", "delete_note"])
def test_every_mutation_runs_only_after_approval(workspace, monkeypatch, operation):
    """逐项覆盖同一个审批工具承载的所有业务分支，验证真实写入而不只检查工具名。"""
    client = TestClient(app); register(client)
    patient = client.post("/api/patients", json={"name": "Before", "date_of_birth": "2000-01-01"}).json()
    pid = patient["id"]
    note = client.post(f"/api/patients/{pid}/notes", json={"content": "Original synthetic note"}).json()
    start = datetime.now(timezone.utc) + timedelta(days=2)
    appointment_data = {"starts_at": start.isoformat(), "ends_at": (start + timedelta(minutes=30)).isoformat(), "reason": "Original visit"}
    client.post("/api/appointments", json={"patient_id": pid, **appointment_data})
    appointment = client.get("/api/appointments").json()[0]
    change = {"operation": operation, "patient_id": pid}
    if operation in {"update_patient", "delete_patient"}:
        change["expected_version"] = client.get(f"/api/patients/{pid}/version").json()["version"]
        if operation == "update_patient":
            change["data"] = {"name": "After", "date_of_birth": "2000-01-01"}
    elif operation in {"update_appointment", "delete_appointment"}:
        change.update(appointment_id=appointment["id"], expected_version=appointment["version"])
        if operation == "update_appointment":
            change["data"] = {**appointment_data, "reason": "Changed visit"}
    elif operation == "create_appointment":
        change["data"] = {"starts_at": (start + timedelta(hours=1)).isoformat(), "ends_at": (start + timedelta(hours=2)).isoformat(), "reason": "New visit"}
    elif operation == "add_note":
        change["content"] = "Added by approved agent"
    else:
        with workspace() as db:
            change.update(note_id=note["id"], expected_version=version(db.get(ClinicalNote, UUID(note["id"])) ))
    original = pms_agent_service.PmsAgentService.build
    monkeypatch.setattr(pms_agent_service.PmsAgentService, "build", lambda self, saver, model=None: original(self, saver, WriteModel(planned_change=change)))
    cid = client.post("/api/conversations", json={"patient_ids": [pid]}).json()["id"]
    run = events(client.post(f"/api/conversations/{cid}/messages", json={"message": "Perform requested operation"}))
    assert run[-1]["type"] == "approval", run
    with workspace() as db:
        assert db.scalar(select(func.count()).select_from(MutationReceipt)) == 0
        assert db.get(Patient, UUID(pid)).name == "Before"
        assert db.scalar(select(func.count()).select_from(ClinicalNote)) == 1
        assert db.scalar(select(func.count()).select_from(Appointment)) == 1
    result = events(client.post(f"/api/conversations/{cid}/resume", json={"interrupt_id": run[-1]["review"]["interrupt_id"], "decisions": [{"type": "approve"}]}))
    assert result[-1]["type"] == "result", result
    with workspace() as db:
        receipt = db.scalar(select(MutationReceipt))
        assert receipt and receipt.operation == operation
        if operation == "update_patient": assert db.get(Patient, UUID(pid)).name == "After"
        if operation == "delete_patient":
            for model in (Patient, ClinicalNote, EmbeddingChunk, Appointment):
                assert db.scalar(select(func.count()).select_from(model)) == 0
        if operation == "create_appointment": assert db.scalar(select(func.count()).select_from(Appointment)) == 2
        if operation == "update_appointment": assert db.get(Appointment, UUID(appointment["id"])).reason == "Changed visit"
        if operation == "delete_appointment": assert db.get(Appointment, UUID(appointment["id"])) is None
        if operation == "add_note":
            assert db.scalar(select(func.count()).select_from(ClinicalNote)) == 2
            assert db.scalar(select(func.count()).select_from(EmbeddingChunk)) == 2
        if operation == "delete_note":
            assert db.get(ClinicalNote, UUID(note["id"])) is None
            assert db.scalar(select(func.count()).select_from(EmbeddingChunk)) == 0


def test_stale_patient_approval_cannot_overwrite_new_note(workspace, monkeypatch):
    client = TestClient(app); register(client)
    patient = client.post("/api/patients", json={"name": "Patient"}).json()
    pid = patient["id"]
    change = {"operation": "delete_patient", "patient_id": pid, "expected_version": client.get(f"/api/patients/{pid}/version").json()["version"]}
    original = pms_agent_service.PmsAgentService.build
    monkeypatch.setattr(pms_agent_service.PmsAgentService, "build", lambda self, saver, model=None: original(self, saver, WriteModel(planned_change=change)))
    cid = client.post("/api/conversations", json={"patient_ids": [pid]}).json()["id"]
    review = events(client.post(f"/api/conversations/{cid}/messages", json={"message": "Delete patient"}))[-1]["review"]
    client.post(f"/api/patients/{pid}/notes", json={"content": "Concurrent new information"})
    result = events(client.post(f"/api/conversations/{cid}/resume", json={"interrupt_id": review["interrupt_id"], "decisions": [{"type": "approve"}]}))
    assert result[-1]["type"] == "result", result
    with workspace() as db:
        assert db.get(Patient, UUID(pid)) is not None
        assert db.scalar(select(func.count()).select_from(MutationReceipt)) == 0
        assert db.scalar(select(func.count()).select_from(ClinicalNote)) == 1


def test_concurrent_appointments_cannot_double_book(workspace):
    """两次并发请求争用同一患者锁，必须一成一败，而非都通过先查后写。"""
    from concurrent.futures import ThreadPoolExecutor
    client = TestClient(app); register(client)
    pid = client.post("/api/patients", json={"name": "Concurrent patient"}).json()["id"]
    start = datetime.now(timezone.utc) + timedelta(days=1)
    data = {"patient_id": pid, "starts_at": start.isoformat(), "ends_at": (start + timedelta(minutes=30)).isoformat(), "reason": "Concurrent visit"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.post, "/api/appointments", json=data) for _ in range(2)]
        assert sorted(f.result().status_code for f in futures) == [201, 409]


def test_plain_provider_reply_is_repaired_and_persisted_through_sse(workspace, monkeypatch):
    """覆盖真实流式路由，不只检查图的 invoke 返回值。"""
    from test_structured_output_recovery import IrregularModel
    provider = IrregularModel(responses=["plain", "structured"])
    original = pms_agent_service.PmsAgentService.build
    monkeypatch.setattr(pms_agent_service.PmsAgentService, "build", lambda self, saver, model=None: original(self, saver, provider))
    client = TestClient(app); register(client)
    cid = client.post("/api/conversations", json={}).json()["id"]
    result = events(client.post(f"/api/conversations/{cid}/messages", json={"message": "Hello"}))
    assert result[-1]["type"] == "result", result
    assert any("纠正" in event["message"] for event in result)
    restored = client.get(f"/api/conversations/{cid}").json()
    assert restored["messages"][-1]["content"] == "Validated answer 2"
    assert restored["can_continue"] is False


def test_legacy_missing_answer_can_continue_without_replaying_writes(workspace, monkeypatch):
    from test_structured_output_recovery import IrregularModel
    from langchain_core.messages import HumanMessage
    provider = IrregularModel(responses=["structured"])
    original = pms_agent_service.PmsAgentService.build
    monkeypatch.setattr(pms_agent_service.PmsAgentService, "build", lambda self, saver, model=None: original(self, saver, provider))
    client = TestClient(app); user = register(client)
    cid = client.post("/api/conversations", json={}).json()["id"]
    with conversation_service.graph_context(UUID(cid), UUID(user["id"])) as (graph, config, service):
        graph.update_state(config, {"messages": [HumanMessage(content="Hello"), AIMessage(content="Legacy reply")], "structured_response": None}, as_node="StructuredOutputMiddleware.after_model")
        assert not graph.get_state(config).next
    assert client.get(f"/api/conversations/{cid}").json()["can_continue"] is True
    result = events(client.post(f"/api/conversations/{cid}/continue"))
    assert result[-1]["type"] == "result", result
    assert provider.tool_sets == [["AgentAnswer"]]
    with workspace() as db:
        assert db.scalar(select(func.count()).select_from(MutationReceipt)) == 0
