"""
agents/planning_agent.py
Selects from the pattern library, customises architecture, decomposes into
atomic testable tasks with risk levels and checkpoint flags.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from core.llm import get_client, parse_json_response
from core.observability import log_action, log_agent_metrics, log_decision
from core.state import ForgeState


def _load_patterns() -> dict:
    lib_path = Path(__file__).parent.parent / "patterns" / "library.json"
    with open(lib_path) as f:
        return json.load(f)


SYSTEM_PROMPT = """You are PlanningAgent. Given a project spec, output a JSON architecture plan.

Output ONLY valid JSON — no explanation, no markdown:
{
  "chosen_pattern": "rest_crud",
  "pattern_rationale": "one sentence",
  "architecture": {
    "overview": "brief description",
    "directory_structure": ["app/", "app/main.py"],
    "key_decisions": [{"decision": "SQLite", "rationale": "simple relational data"}],
    "data_models": [{"name": "Item", "fields": [{"name": "id", "type": "int"}, {"name": "name", "type": "str"}, {"name": "price", "type": "float"}]}],
    "api_endpoints": [{"method": "GET", "path": "/items", "description": "list all items, returns list of ItemResponse"}, {"method": "POST", "path": "/items", "description": "create item, body: ItemCreate, returns ItemResponse with 201"}]
  },
  "task_plan": [
    {
      "id": "task_001",
      "title": "Database setup",
      "description": "Create app/database.py: define Base(DeclarativeBase), create_engine with DATABASE_URL env var (default sqlite:///./app.db), SessionLocal = sessionmaker, get_db() generator that yields Session, init_db() that calls Base.metadata.create_all(engine). No models here — only the Base and session infrastructure.",
      "files": ["app/database.py"],
      "test_file": "tests/test_database.py",
      "risk_level": "low",
      "checkpoint_flag": true,
      "depends_on": []
    },
    {
      "id": "task_002",
      "title": "Item SQLAlchemy model",
      "description": "Create app/models.py: import Base from app.database. Define Item(Base) with __tablename__='items', id: Mapped[int] primary key, name: Mapped[str] not null, price: Mapped[float], created_at: Mapped[datetime] default=utcnow. Use SQLAlchemy 2.x mapped_column syntax.",
      "files": ["app/models.py"],
      "test_file": "tests/test_models.py",
      "risk_level": "low",
      "checkpoint_flag": false,
      "depends_on": ["task_001"]
    }
  ],
  "confidence": 85,
  "confidence_reasoning": "one sentence",
  "alternatives_considered": ["PostgreSQL rejected — SQLite sufficient"]
}

TASK DESCRIPTION RULES (critical — descriptions drive code generation):
- Every description must name the EXACT file, class names, function names, and field names to create.
- Describe WHAT the code must do, not HOW to implement it in general terms.
- For database tasks: name the Base class, session factory, and dependency function.
- For model tasks: list every field with its type and constraints.
- For router tasks: list every endpoint with method, path, request body type, and response type.
- For auth tasks: specify the JWT algorithm, token fields, and the dependency function signature.
- Bad description: "Set up the database". Good: "Create database.py with Base, engine, get_db(), init_db()".

TASK STRUCTURE RULES:
- Each task must have 1-3 files max. Never put all files in one task.
- Order tasks so dependencies come first (database → models → schemas → routers → main).
- Set checkpoint_flag=true for the first task, any high-risk task, and the final task.
- risk_level: low=standard patterns, medium=some complexity, high=security/integration/novel.
- Every task must have a test_file (TDD-first is mandatory)."""


def run(state: ForgeState) -> ForgeState:
    t0 = time.time()
    log_action("PlanningAgent", "Starting architecture planning")

    patterns = _load_patterns()
    spec = state.get("structured_spec", {})
    tier = state.get("complexity_tier", 1)
    answers = state.get("clarifying_answers", {})

    # Auto-select pattern based on tier
    tier_key = f"tier_{tier}"
    suggested_pattern = patterns["pattern_selection_rules"].get(tier_key, "rest_crud")
    pattern_detail = patterns["patterns"].get(suggested_pattern, {})

    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Project summary: {spec.get('project_summary','')[:400]}\n"
                f"Complexity tier: {tier}\n"
                f"Stack: {json.dumps(spec.get('inferred_stack', {}))}\n"
                f"Acceptance criteria: {json.dumps(spec.get('acceptance_criteria', []))}\n"
                f"User answers: {json.dumps(answers)}\n"
                f"Priority weights: {json.dumps(state.get('priority_weights', {}))}\n"
                f"Suggested pattern: {suggested_pattern}\n"
                f"Pattern tasks: {json.dumps(pattern_detail.get('task_templates', []))}\n"
                f"Pattern structure: {json.dumps(pattern_detail.get('directory_structure', []))}\n\n"
                f"IMPORTANT: Task descriptions must name exact class names, function names, and "
                f"field names — not just general goals. The code generator reads descriptions "
                f"literally to know what to implement.\n"
                f"Order tasks so dependencies come first: database -> models -> schemas -> routers -> main.\n"
                "Output the architecture and task plan JSON."
            ),
        },
    ]

    response, usage = client.call_reasoning(
        messages, agent_name="PlanningAgent", max_tokens=3500
    )

    try:
        plan = parse_json_response(response)
    except ValueError as e:
        log_action("PlanningAgent", "JSON parse failed, retrying with stricter prompt", str(e), "WARN")
        # On retry: don't append the full broken response (token waste).
        # Instead send a fresh, tighter prompt asking for the plan only.
        retry_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Project: {spec.get('project_summary','')[:200]}\n"
                    f"Tier: {tier}, Pattern: {suggested_pattern}\n"
                    f"Stack: {json.dumps(spec.get('inferred_stack', {}))}\n\n"
                    "Output ONLY compact JSON with chosen_pattern, architecture "
                    "(overview + key_decisions only), and task_plan (max 6 tasks, "
                    "each with id/title/description/files/test_file/risk_level/"
                    "checkpoint_flag). No directory_structure, no data_models, "
                    "no api_endpoints — keep it short."
                ),
            },
        ]
        response, usage2 = client.call_reasoning(
            retry_messages, agent_name="PlanningAgent", max_tokens=2500
        )
        usage["total_tokens"] += usage2["total_tokens"]
        plan = parse_json_response(response)

    latency = time.time() - t0
    confidence = plan.get("confidence", 75)

    # Normalise task_plan to TaskItem format
    task_plan = []
    for task in plan.get("task_plan", []):
        task_plan.append({
            "id": task.get("id", f"task_{len(task_plan):03d}"),
            "title": task.get("title", ""),
            "description": task.get("description", ""),
            "risk_level": task.get("risk_level", "medium"),
            "checkpoint_flag": task.get("checkpoint_flag", False),
            "status": "pending",
            "files": task.get("files", []),
            "test_file": task.get("test_file"),
        })

    decision = {
        "agent": "PlanningAgent",
        "confidence": confidence,
        "reasoning": plan.get("confidence_reasoning", ""),
        "alternatives": plan.get("alternatives_considered", []),
        "outcome": None,
    }

    metrics = {
        "latency_s": round(latency, 3),
        "tokens": usage["total_tokens"],
        "confidence": confidence,
        "tasks_planned": len(task_plan),
        "pattern": plan.get("chosen_pattern"),
    }
    log_agent_metrics("PlanningAgent", metrics)
    log_decision(decision)

    log_action(
        "PlanningAgent",
        "Planning complete",
        f"pattern={plan.get('chosen_pattern')} tasks={len(task_plan)} confidence={confidence}%",
    )

    state["chosen_pattern"] = plan.get("chosen_pattern", suggested_pattern)
    state["architecture"] = plan.get("architecture", {})
    state["task_plan"] = task_plan
    state["current_confidence"] = confidence
    state["decision_audit"] = state.get("decision_audit", []) + [decision]
    state["agent_metrics"]["PlanningAgent"] = metrics
    state["api_call_count"] = state.get("api_call_count", 0) + 1
    state["total_tokens"] = state.get("total_tokens", 0) + usage["total_tokens"]
    state["phase"] = "planning_done"
    state["user_approved_plan"] = False

    return state