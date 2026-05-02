"""
tests/test_students_router.py
TDD-first tests for app/routers/students.py (CRUD endpoints)
Written by Forge QAAgent BEFORE implementation files exist.
"""
import pytest

try:
    from app.main import app
    from starlette.testclient import TestClient
    HAS_APP = True
except ImportError:
    HAS_APP = False


@pytest.fixture(scope="module")
def client():
    if not HAS_APP:
        pytest.skip("app not yet implemented")
    with TestClient(app) as c:
        yield c


def test_create_student_valid_returns_201(client):
    payload = {"name": "Alice Smith", "email": "alice@test.com", "grade": "A"}
    response = client.post("/students", json=payload)
    assert response.status_code in (200, 201)
    data = response.json()
    assert "id" in data
    assert data["email"] == "alice@test.com"


def test_list_students_returns_200(client):
    response = client.get("/students")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_student_by_id_returns_correct_record(client):
    create_resp = client.post("/students", json={"name": "Bob", "email": "bob@test.com", "grade": "B"})
    assert create_resp.status_code in (200, 201)
    student_id = create_resp.json()["id"]
    get_resp = client.get(f"/students/{student_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == student_id


def test_get_nonexistent_student_returns_404(client):
    response = client.get("/students/99999")
    assert response.status_code == 404


def test_update_student_returns_updated_record(client):
    create_resp = client.post("/students", json={"name": "Carol", "email": "carol@test.com", "grade": "C"})
    student_id = create_resp.json()["id"]
    update_resp = client.put(f"/students/{student_id}", json={"name": "Carol Updated", "email": "carol@test.com", "grade": "A"})
    assert update_resp.status_code == 200
    assert update_resp.json()["name"] == "Carol Updated"


def test_delete_student_returns_204(client):
    create_resp = client.post("/students", json={"name": "Delete Me", "email": "delete@test.com", "grade": "D"})
    student_id = create_resp.json()["id"]
    del_resp = client.delete(f"/students/{student_id}")
    assert del_resp.status_code in (200, 204)
    get_resp = client.get(f"/students/{student_id}")
    assert get_resp.status_code == 404


def test_create_student_invalid_email_returns_422(client):
    response = client.post("/students", json={"name": "Bad Email", "email": "not-an-email", "grade": "A"})
    assert response.status_code == 422


def test_create_student_missing_required_field_returns_422(client):
    response = client.post("/students", json={"name": "No Email"})
    assert response.status_code == 422
