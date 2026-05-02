"""
agents/qa_agent.py
Writes failing pytest test cases BEFORE production code is generated.
TDD-first is non-negotiable. Tests define the acceptance contract.
"""
from __future__ import annotations
import json, time
from core.llm import get_client
from core.observability import log_action, log_agent_metrics
from core.state import ForgeState, TaskItem


SYSTEM_PROMPT = """You are the QAAgent for Forge. Write pytest tests for the given task.

CRITICAL RULES:
1. Output ONLY valid Python code. No markdown. No explanation.
2. Always start with: import pytest
3. For FastAPI apps: use TestClient from starlette.testclient, import the app from its module.
4. For SQLAlchemy models: use an in-memory SQLite database in fixtures, never assume the DB exists.
5. Write a pytest fixture for the test client and database if needed.
6. Test HTTP status codes and response structure — not exact field values.
7. Use pytest.mark.parametrize for multiple similar cases.
8. Every test function name: test_<what>_<condition>_returns_<expected>.
9. Do NOT import things that don't exist yet — only import from files listed in the task's files list.
10. Write 3-5 focused test functions. Quality over quantity.
11. Wrap ALL imports in try/except ImportError with pytest.skip() — the files don't exist yet when
    tests are written (TDD-first means tests are written before implementation).
12. Use conftest.py fixtures when available (client, db, session).
13. NEVER hardcode expected IDs or timestamps — test structure and status codes only.

IMPORT SAFETY PATTERN (always use this):
    import pytest
    try:
        from app.main import app
        from starlette.testclient import TestClient
        HAS_APP = True
    except ImportError:
        HAS_APP = False

    @pytest.fixture
    def client():
        if not HAS_APP:
            pytest.skip("app not yet implemented")
        return TestClient(app)

FASTAPI ROUTE TEST TEMPLATE:
    def test_create_item_valid_returns_201(client):
        response = client.post("/items", json={"name": "test", "value": 1.0})
        assert response.status_code in (200, 201)
        data = response.json()
        assert "id" in data or "name" in data

SQLALCHEMY MODEL TEST TEMPLATE:
    @pytest.fixture
    def db():
        try:
            from app.database import Base
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
        except ImportError:
            pytest.skip("database not yet implemented")
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

JWT AUTH TEST TEMPLATE:
    def test_login_valid_credentials_returns_token(client):
        response = client.post("/auth/token",
            data={"username": "testuser", "password": "testpass"})
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_protected_route_without_token_returns_401(client):
        response = client.get("/protected-route")
        assert response.status_code == 401

MONGODB TEST TEMPLATE:
    @pytest.fixture
    def mock_collection(monkeypatch):
        # Mock MongoDB collection for unit testing without a real DB
        class FakeCollection:
            async def find_one(self, query): return None
            async def insert_one(self, doc): return type("r", (), {"inserted_id": "fake_id"})()
            async def find(self, query=None): return iter([])
        return FakeCollection()"""


# Tier-specific test focus areas — guides what the QA agent emphasises
TIER_TEST_FOCUS = {
    1: "CRUD operations, schema validation, 404 on missing, 422 on invalid input",
    2: "business rule correctness, edge cases, boundary values, invalid rule configs",
    3: "async HTTP calls, timeout handling, retry behaviour, error responses from 3rd party",
    4: "JWT token issuance, token expiry, RBAC role enforcement, 401/403 responses",
    5: "MongoDB aggregation correctness, change stream events, join result shape, sync completeness",
}


def run(state: ForgeState) -> ForgeState:
    log_action("QAAgent", "Writing TDD test suite")

    task_plan    = state.get("task_plan", [])
    architecture = state.get("architecture", {})
    spec         = state.get("structured_spec", {})
    test_files   = state.get("test_files", {})
    tier         = state.get("complexity_tier", 1)

    client = get_client()
    tasks_with_tests = 0

    # Build a map of all project files for safe import guidance
    all_project_files = {fp for task in task_plan for fp in task.get("files", [])}

    for i, task in enumerate(task_plan):
        if task.get("test_file") is None:
            continue

        log_action("QAAgent", f"Writing tests for: {task['title']}")

        task_files      = task.get("files", [])
        task_file_mods  = [fp.replace("/", ".").removesuffix(".py") for fp in task_files
                           if fp.endswith(".py")]
        tier_focus      = TIER_TEST_FOCUS.get(tier, TIER_TEST_FOCUS[1])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Write pytest tests for this task.\n\n"
                    f"Task title: {task['title']}\n"
                    f"Task description: {task['description'][:400]}\n"
                    f"Complexity tier: {tier} — focus on: {tier_focus}\n"
                    f"Test file to write: {task['test_file']}\n\n"
                    f"Files this task produces (ONLY import from these):\n"
                    + "\n".join(f"  - {mod}" for mod in task_file_mods) + "\n\n"
                    f"API endpoints:\n"
                    f"{json.dumps(architecture.get('api_endpoints', [])[:6], indent=2)}\n\n"
                    f"Data models:\n"
                    f"{json.dumps(architecture.get('data_models', [])[:3], indent=2)}\n\n"
                    f"IMPORT SAFETY: Wrap all imports in try/except ImportError + pytest.skip().\n"
                    f"This is TDD — the implementation files do NOT exist yet when you write tests.\n\n"
                    "Write 3-5 focused test functions. "
                    "Test status codes and response shape, not exact values. "
                    "Use the import safety pattern shown in the system prompt."
                ),
            },
        ]

        t0 = time.time()
        response, usage = client.call(messages, agent_name="QAAgent")
        latency = time.time() - t0

        # Clean up response
        test_code = response.strip()
        if test_code.startswith("```"):
            lines = test_code.split("\n")
            end = len(lines) - 1
            while end > 0 and lines[end].strip().startswith("```"):
                end -= 1
            test_code = "\n".join(lines[1:end + 1])

        # Validate the test code is syntactically correct before storing
        # A broken test file is worse than no test file
        if _is_valid_python(test_code):
            test_files[task["test_file"]] = test_code
            tasks_with_tests += 1
        else:
            # Ask for a simpler version on syntax failure
            log_action("QAAgent", f"  Test syntax error — retrying simplified for {task['test_file']}", level="WARN")
            simple_response, usage2 = client.call(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Write 2 simple pytest tests for: {task['title']}\n"
                        f"Import from: {task_file_mods[0] if task_file_mods else 'app.main'}\n"
                        f"Use try/except ImportError + pytest.skip() around all imports.\n"
                        f"Output ONLY raw Python. No markdown."
                    )},
                ],
                agent_name="QAAgent",
            )
            usage["total_tokens"] += usage2["total_tokens"]
            simple_code = simple_response.strip()
            if simple_code.startswith("```"):
                lines = simple_code.split("\n")
                simple_code = "\n".join(lines[1:-1])
            test_files[task["test_file"]] = simple_code
            tasks_with_tests += 1

        state["api_call_count"] = state.get("api_call_count", 0) + 1
        state["total_tokens"]   = state.get("total_tokens", 0) + usage["total_tokens"]

        log_action(
            "QAAgent",
            f"Tests written for task {task['id']}",
            f"file={task['test_file']} latency={latency:.1f}s",
        )

    metrics = {
        "tasks_with_tests":      tasks_with_tests,
        "test_files_generated":  len(test_files),
    }
    log_agent_metrics("QAAgent", metrics)
    log_action("QAAgent", f"TDD suite complete — {tasks_with_tests} test files written")

    state["test_files"]             = test_files
    state["agent_metrics"]["QAAgent"] = metrics
    state["phase"]                  = "tests_written"
    return state


def _is_valid_python(code: str) -> bool:
    """Quick syntax check — returns True if the code parses without error."""
    try:
        compile(code, "<test>", "exec")
        return True
    except SyntaxError:
        return False