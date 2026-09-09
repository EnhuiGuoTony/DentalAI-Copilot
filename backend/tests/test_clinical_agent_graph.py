from uuid import uuid4

from app.db.models import ClinicalCase
from app.services.clinical_agent_graph import ClinicalAgentGraph


class EmptyResult:
    def scalars(self):
        return self

    def all(self):
        return []


class FakeDb:
    def execute(self, _statement):
        return EmptyResult()


def test_langgraph_agent_runs_with_mock_llm():
    case = ClinicalCase(id=uuid4(), patient_id=uuid4(), title="Demo case")
    graph = ClinicalAgentGraph(FakeDb())

    result = graph.run(case, "Summarize this case")

    assert "output" in result
    assert result["output"].clinical_summary
    assert [item.tool for item in result["tool_trace"]] == [
        "langgraph.load_xray_findings",
        "langgraph.search_patient_history",
        "langgraph.get_structured_clinical_facts",
        "langgraph.generate_clinical_draft",
    ]
