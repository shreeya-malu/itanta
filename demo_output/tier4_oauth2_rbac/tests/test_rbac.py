"""
tests/test_rbac.py
TDD-first tests for Role-Based Access Control enforcement.
Tier 4 — The Gatekeeper. Written by Forge QAAgent BEFORE implementation.
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


@pytest.mark.parametrize("role,path,expected_allowed", [
    ("user",  "/admin/users",    False),
    ("admin", "/admin/users",    True),
    ("user",  "/users/me",       True),
    ("admin", "/users/me",       True),
])
def test_rbac_role_enforcement(client, role, path, expected_allowed):
    try:
        from jose import jwt
        import os
        from datetime import datetime, timedelta
        secret = os.getenv("SECRET_KEY", "test-secret-key")
        token = jwt.encode(
            {"sub": f"{role}user", "role": role, "exp": datetime.utcnow() + timedelta(hours=1)},
            secret, algorithm="HS256"
        )
    except ImportError:
        pytest.skip("jose not installed")

    response = client.get(path, headers={"Authorization": f"Bearer {token}"})
    if expected_allowed:
        assert response.status_code not in (401, 403), \
            f"Role '{role}' should have access to {path}, got {response.status_code}"
    else:
        assert response.status_code in (401, 403), \
            f"Role '{role}' should be denied access to {path}, got {response.status_code}"


def test_no_token_blocked_on_all_protected_routes(client):
    for path in ["/users/me", "/admin/users"]:
        response = client.get(path)
        assert response.status_code == 401, f"Expected 401 on {path} without token"
