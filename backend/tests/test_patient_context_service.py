from app.services.patient_context_service import PatientContextService


def test_mock_answer_reports_exact_empty_database_counts():
    service = PatientContextService.__new__(PatientContextService)
    answer = service.mock_answer(
        {
            "statistics": {"patients": 0, "cases": 0, "notes": 0, "images": 0, "findings": 0, "clinical_facts": 0},
            "patients": [],
        }
    )

    assert "病人：0 位" in answer
    assert "尚未导入或创建任何病人" in answer


def test_mock_answer_includes_patient_chart_summary():
    service = PatientContextService.__new__(PatientContextService)
    answer = service.mock_answer(
        {
            "statistics": {"patients": 1, "cases": 2, "notes": 3, "images": 4, "findings": 5, "clinical_facts": 6},
            "patients": [{"name": "测试病人", "case_count": 2, "note_count": 3, "image_count": 4}],
        }
    )

    assert "病人：1 位" in answer
    assert "测试病人（病例 2，笔记 3，影像 4）" in answer


def test_patient_statistics_questions_bypass_the_model():
    assert PatientContextService.requires_deterministic_answer("有多少个病人？")
    assert PatientContextService.requires_deterministic_answer("List all patients")
    assert not PatientContextService.requires_deterministic_answer("龋齿通常如何处理？")
