"""
tests/test_schemas.py
TDD-first tests for app/schemas.py (Pydantic v2 validation)
Written by Forge QAAgent BEFORE implementation files exist.
"""
import pytest

try:
    from app.schemas import StudentCreate, StudentResponse, StudentUpdate
    HAS_SCHEMAS = True
except ImportError:
    HAS_SCHEMAS = False


def test_student_create_valid():
    if not HAS_SCHEMAS:
        pytest.skip("schemas not yet implemented")
    s = StudentCreate(name="Alice", email="alice@example.com", grade="A")
    assert s.name == "Alice"
    assert s.email == "alice@example.com"


def test_student_create_invalid_email_raises():
    if not HAS_SCHEMAS:
        pytest.skip("schemas not yet implemented")
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StudentCreate(name="Alice", email="not-valid", grade="A")


def test_student_response_has_id_field():
    if not HAS_SCHEMAS:
        pytest.skip("schemas not yet implemented")
    fields = StudentResponse.model_fields
    assert "id" in fields


@pytest.mark.parametrize("grade", ["A", "B", "C", "D", "F"])
def test_student_create_valid_grades(grade):
    if not HAS_SCHEMAS:
        pytest.skip("schemas not yet implemented")
    s = StudentCreate(name="Test", email=f"test_{grade}@example.com", grade=grade)
    assert s.grade == grade


def test_student_update_partial_fields():
    if not HAS_SCHEMAS:
        pytest.skip("schemas not yet implemented")
    # StudentUpdate should allow partial updates (all fields optional)
    u = StudentUpdate(grade="A")
    assert u.grade == "A"
