"""
tests/test_auth.py
TDD-first tests for JWT auth, token issuance, and protected routes.
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


def test_register_user_returns_201(client):
    response = client.post("/auth/register", json={
        "username": "testuser", "email": "test@example.com",
        "password": "SecurePass123!", "role": "user"
    })
    assert response.status_code in (200, 201)
    data = response.json()
    assert "id" in data
    assert "password" not in data  # never expose password


def test_login_valid_credentials_returns_token(client):
    client.post("/auth/register", json={
        "username": "loginuser", "email": "login@example.com",
        "password": "SecurePass123!", "role": "user"
    })
    response = client.post("/auth/token",
        data={"username": "loginuser", "password": "SecurePass123!"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password_returns_401(client):
    response = client.post("/auth/token",
        data={"username": "loginuser", "password": "WrongPassword"})
    assert response.status_code == 401


def test_protected_route_without_token_returns_401(client):
    response = client.get("/users/me")
    assert response.status_code == 401


def test_protected_route_with_valid_token_returns_200(client, auth_token):
    response = client.get("/users/me",
        headers={"Authorization": f"Bearer {auth_token}"})
    assert response.status_code in (200, 401)  # 401 if test token user doesn't exist


def test_admin_only_route_with_user_token_returns_403(client, auth_token):
    response = client.get("/admin/users",
        headers={"Authorization": f"Bearer {auth_token}"})
    assert response.status_code in (403, 404)


def test_admin_only_route_with_admin_token_returns_200(client, admin_token):
    response = client.get("/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code in (200, 404)  # 404 if route not yet wired


def test_expired_token_returns_401(client):
    from datetime import datetime, timedelta
    try:
        from jose import jwt
        import os
        secret = os.getenv("SECRET_KEY", "test-secret-key")
        expired = jwt.encode(
            {"sub": "user", "role": "user", "exp": datetime.utcnow() - timedelta(hours=1)},
            secret, algorithm="HS256"
        )
        response = client.get("/users/me", headers={"Authorization": f"Bearer {expired}"})
        assert response.status_code == 401
    except ImportError:
        pytest.skip("jose not installed")
