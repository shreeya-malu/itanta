# FORGE — Engineering Intelligence System

> From natural-language prompt → fully-verified, Dockerized software  
> with transparent reasoning, confidence-aware agents, and measurable decision intelligence.

**Itanta AI Hackathon · Team BrainBit · Agentic AI Systems Track**

---

## Demo

▶ [Watch the full pipeline demo (Loom)]https://www.loom.com/share/554b6147fcf04fc9ba8b3bb1aca3f9c1

The demo shows Forge accepting the prompt `"Build OAuth2 + JWT authentication with RBAC. Roles: admin, editor, viewer. bcrypt passwords. RS256 JWT, 15-min expiry. Registration, login, refresh, logout. Admin role management."` and autonomously producing a working, tested, Dockerized FastAPI application in ~2.5 minutes.

---

## Implementation Note: Model Selection

The original design proposal specified **Phi-3 Mini + DeepSeek Coder 6.7B via Ollama** for fully local execution. During implementation, this was deliberately evolved to use **Groq-hosted Llama 3.3 70B** (`llama-3.3-70b-versatile`) for the following reasons:

| Factor | Ollama (proposed) | Groq (implemented) |
|--------|------------------|--------------------|
| Model quality | Phi-3 Mini (3.8B) | Llama 3.3 70B (70B) |
| Inference speed | ~2–8 tokens/s on consumer hardware | ~200+ tokens/s |
| RAM requirement | 8GB | None (cloud inference) |
| Reliability | Varies by hardware | Consistent |
| Cost | Free (local) | Free (Groq free tier) |
| Colab compatibility | Requires Ollama install | Single API key |

**Groq's free tier is functionally equivalent to local execution for judging purposes** — no credit card, no billing. Switching back to Ollama requires only changing `provider: groq` to `provider: ollama` in `config.yaml` and swapping the client in `core/llm.py`. The architecture is provider-agnostic by design.

---

## Quick Start — Google Colab (Recommended)

1. Open `Forge_Launcher.ipynb` in Google Colab
2. Add your **`GROQ_API_KEY`** via left sidebar → Secrets ([get a free key](https://console.groq.com))
3. Optionally add `WANDB_API_KEY` and `LANGSMITH_API_KEY` (both free, both optional)
4. Run all cells top to bottom (~3 min to install)
5. Upload your **`forge.zip`** when prompted
6. Launch the dashboard or run CLI

---

## Local Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone / extract the project
cd forge/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set required environment variables
export GROQ_API_KEY="your-key-here"          # Required — get free at console.groq.com
export WANDB_API_KEY="your-key"              # Optional — wandb.ai/authorize
export LANGSMITH_API_KEY="your-key"          # Optional — smith.langchain.com

# 4a. Launch the Gradio dashboard (recommended)
python main.py --mode dashboard

# 4b. OR run headless CLI
python main.py --mode cli --prompt "Build a REST API for managing student records with authentication"

# 5. Run tests on a generated project
cd generated_project/
pytest ../tests/ -v
```

---

## Project Structure

```
forge/
├── Forge_Launcher.ipynb      ← Colab entry point
├── main.py                   ← CLI entry point
├── config.yaml               ← All configurable settings
├── requirements.txt
├── pytest.ini                ← Test discovery config
│
├── agents/
│   ├── requirement_agent.py      ← Parses prompt → structured JSON spec
│   ├── planning_agent.py         ← Architecture + task decomposition (merged TaskDecompositionAgent)
│   ├── architecture_validator.py ← Cross-checks architecture against spec
│   ├── qa_agent.py               ← TDD-first test generation (tests before code)
│   ├── codegen_agent.py          ← File-by-file code generation with per-file contracts
│   ├── debug_agent.py            ← Targeted error fixing with retry loop
│   ├── security_audit_agent.py   ← bandit scanning + immediate remediation
│   └── docker_agent.py           ← Template-based Dockerfile + docker-compose
│
├── core/
│   ├── state.py              ← LangGraph TypedDict state schema
│   ├── graph.py              ← Pipeline graph + conditional routing
│   ├── llm.py                ← Groq client wrapper with smart token trimming
│   ├── observability.py      ← W&B + LangSmith + activity log
│   └── validators.py         ← ruff, pylint, bandit, pytest runners
│
├── dashboard/
│   └── app.py                ← Gradio real-time dashboard (11 tabs)
│
├── patterns/
│   └── library.json          ← Architecture pattern library
│
└── demo_output/              ← Pre-run pipeline outputs for inspection
    ├── tier1_student_records/    ← Tier 1: Basic CRUD API (complete output)
    │   ├── tests/                ← TDD-first test suite (written before code)
    │   ├── outputs/summary.json
    │   └── pytest.ini
    └── tier4_oauth2_rbac/        ← Tier 4: OAuth2 + JWT + RBAC (complete output)
        ├── tests/
        ├── outputs/summary.json
        └── pytest.ini
```

### Design Note: Task Decomposition Agent

The original proposal listed `TaskDecompositionAgent` as a separate agent. During implementation, this was consolidated into `PlanningAgent`, which now outputs both the architecture and the full `task_plan` in a single structured JSON response. This is a better design: it eliminates a round-trip LLM call and ensures the task plan is always consistent with the architecture that produced it. A separate decomposition agent adds latency without adding reasoning quality.

---

## Agent Pipeline

```
RequirementAgent → PlanningAgent → ArchitectureValidatorAgent
    → QAAgent → CodeGenAgent → DebugAgent → SecurityAuditAgent
    → SecurityFixAgent → DockerAgent → Summary
```

### TDD-First Approach

`QAAgent` runs **before** `CodeGenAgent`. Tests are written against a contract (the task description and architecture spec), not against existing code. This forces the generated implementation to satisfy the test contract, not the other way around.

Every test file uses the import-safety pattern:

```python
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
```

This means `pytest` can collect and report tests even when run against an empty project — which is essential for demonstrating the TDD timeline.

---

## Confidence Routing

| Score | Behaviour |
|-------|-----------|
| ≥ 80% | Proceed autonomously |
| 60–79% | Flag in dashboard, proceed with warning |
| < 60%  | Pause — await human clarification |

Confidence is computed per agent and logged to the Decision Audit Trail (see dashboard tab below).

---

## Failure Recovery

When generated code fails validation, `DebugAgent` runs a targeted fix loop:

1. Parses ruff / pylint / pytest error output
2. Identifies the minimum set of files to change
3. Re-generates only affected sections using a per-file contract
4. Retries up to 3 times (configurable in `config.yaml`)
5. On exhaustion: returns partial working implementation with a failure report

---

## Decision Audit Trail

Every agent records a structured decision entry in `state["decision_audit"]`:

```json
{
  "agent": "PlanningAgent",
  "confidence": 87,
  "reasoning": "rest_crud pattern directly matches requirements",
  "alternatives": [
    "PostgreSQL rejected — SQLite sufficient for single-instance",
    "Django rejected — FastAPI preferred for Pydantic v2 integration"
  ],
  "outcome": "approved"
}
```

This gives judges and users full visibility into **why** each architectural choice was made and what alternatives were considered and rejected. It is displayed live in the **Decision Audit Trail** tab and written to `outputs/summary.json` on completion.

---

## Dashboard Guide

Launch with `python main.py --mode dashboard` or via the Colab notebook. The Gradio UI is available at `http://localhost:7860` (or the public share URL printed in the console).

### Main Controls

**Project Specification** — the prompt input field. Type your natural-language project idea here. The Run button activates once text is present.

**Run Forge** — starts the full pipeline. A background thread runs the LangGraph graph while the dashboard polls and updates every 2 seconds.

**Stop** — gracefully halts the pipeline between agent steps.

**Priority Weights (optional)** — expand to adjust 5 sliders that bias agent decisions:

| Slider | Effect |
|--------|--------|
| Generation Speed | Prefers fewer files, simpler patterns |
| Code Quality | Enables stricter linting, more detailed contracts |
| Test Coverage | More test functions per task, stricter coverage |
| Security Hardness | More aggressive bandit checks, forces fixes |
| Simplicity | Avoids over-engineering, prefers single-file modules |

Weights are passed into `RequirementAgent`'s prompt and influence the `priority_weights` field of the structured spec.

---

### Dashboard Tabs

#### Live Metrics

Real-time pipeline statistics updated every 2 seconds while the pipeline runs:

- **API Calls** — total Groq API calls made across all agents
- **Total Tokens** — cumulative token spend (useful for staying within Groq free tier limits)
- **Current Phase** — which pipeline stage is active (e.g. `tests_written`, `codegen_done`)
- **Status** — `RUNNING` / `WAITING` / `COMPLETED` / `FAILED`
- **Per-agent table** — latency in seconds, tokens used, and confidence score for each completed agent

Use this tab to monitor pipeline health in real time and spot slow or expensive agents.

#### Decision Audit Trail

Shows every structured decision recorded by agents — what was decided, at what confidence level, what the reasoning was, and what alternatives were explicitly rejected. Updates live as agents complete.

This is Forge's core differentiator for the "systems thinking" requirement: the pipeline doesn't just generate code, it records an auditable chain of reasoning.

Color coding: green ≥ 80%, amber 60–79%, red < 60%.

#### Task Plan

The ordered list of atomic engineering tasks generated by `PlanningAgent`. Each row shows:

- Task ID and title
- Status: Pending → In Progress → Done / Failed
- Risk level: LOW / MEDIUM / HIGH

Watch this tab to see tasks move from pending to done in real time as `CodeGenAgent` works through them sequentially.

#### Test Results

Per-file pytest results for every test file in the generated project. Shows pass/fail counts and which files passed. Updated after `DebugAgent` runs. All tests are written by `QAAgent` **before** `CodeGenAgent` runs — this tab proves the TDD-first contract is satisfied.

#### Security

Bandit security audit findings for every generated Python file. Shows:

- File and line number
- Severity: HIGH / MEDIUM / LOW
- Issue description
- Whether `SecurityFixAgent` automatically applied a fix (Fixed / Open)

False-positive codes are pre-filtered: `B101` (assert in tests) and `B104` (bind all interfaces in dev server) are suppressed.

#### Architecture

The structured architecture specification produced by `PlanningAgent` and validated by `ArchitectureValidatorAgent`. Displays:

- Technology stack decisions and their rationale
- Module structure
- API endpoint list with methods and paths
- Data model schemas
- Validation verdict (Approved / Approved with warnings / Requires revision)

If `ArchitectureValidatorAgent` returns `requires_revision`, the pipeline routes back to `PlanningAgent` automatically.

#### Files & Editor

A live file browser for all generated source files. Features:

- **File selector dropdown** — browse every generated file (`.py`, `Dockerfile`, `requirements.txt`, `.env.example`, etc.)
- **Editable text area** — view and manually edit any file in the browser
- **Save changes** — write edits back to the in-memory state
- **AI Edit** — describe a change in plain English (e.g. `"Add a /students/summary endpoint that returns grade distribution"`) and `CodeGenAgent` applies it to the selected file automatically
- **⬇ Download ZIP** — download the entire generated project as a zip archive

This tab is the primary way to inspect what Forge produced and make targeted modifications without re-running the full pipeline.

#### Project Preview

A rich visual summary of the generated project, built by parsing the actual generated code (not just the architecture plan):

- **Stats bar** — file count, lines of code, endpoint count, tasks completed, test pass rate
- **API Endpoint Explorer** — every detected route with method badge, path, description, and an expandable section showing realistic sample request body, expected response, curl command, and Python `requests` snippet
- **Data Models** — field-by-field schema tables parsed from SQLAlchemy models
- **File Tree** — all files with line counts and role icons (main, database, auth, tests...)
- **Run instructions** — project-specific startup commands derived from the actual stack (e.g. `alembic upgrade head` only appears if Alembic is in the generated code)
- **Docker section** — `docker compose up --build` with the correct port

This tab is designed for demo purposes — it gives a judge or stakeholder a full picture of the project without reading raw code.

#### Summary

Final pipeline report after completion. Shows:

- Total tasks completed vs. failed
- Test pass rate
- Security findings and fixes
- Full Decision Audit Trail
- Token and API call totals
- Links to output artifacts (`outputs/summary.json`, `outputs/security_report.json`, `outputs/specification.json`)

---

## Complexity Tiers

Forge classifies each project prompt into one of 5 complexity tiers, selecting the appropriate architecture pattern and adjusting agent behaviour accordingly.

| Tier | Name | Core Challenge | Example Prompt |
|------|------|---------------|----------------|
| 1 | The Ledger | Basic CRUD with strict schema validation | "Student records API with authentication" |
| 2 | Logic Engine | Dynamic business rules engine | "Loan eligibility calculator with configurable rules" |
| 3 | Live Bridge | Async 3rd-party API integration | "Weather aggregator that polls 3 external APIs" |
| 4 | The Gatekeeper | OAuth2 + JWT + RBAC | "Secure API with user/admin roles and protected routes" |
| 5 | Mongo-SQL Engine | MongoDB aggregation + Change Streams | "Analytics API bridging MongoDB events to PostgreSQL" |

See `demo_output/` for pre-run examples of Tier 1 and Tier 4.

### How tier affects agent behaviour

- **QAAgent** uses `TIER_TEST_FOCUS` to emphasise the right test categories (e.g. Tier 4 focuses on `JWT token issuance, token expiry, RBAC role enforcement, 401/403 responses`)
- **PlanningAgent** selects from `patterns/library.json` based on tier
- **DebugAgent** gets additional context about expected failure modes per tier
- **SecurityAuditAgent** applies stricter checks at Tier 3+ (async patterns, credential handling)

---

## Configuration

All settings in `config.yaml`:

```yaml
llm:
  provider: groq
  planning_model: llama-3.3-70b-versatile   # reasoning / planning
  codegen_model: llama-3.3-70b-versatile    # code generation
  fast_model: llama-3.1-8b-instant          # boilerplate, configs
  temperature_codegen: 0.15                 # low = deterministic code

pipeline:
  max_debug_retries: 3
  max_security_fix_retries: 3
  confidence_autonomous: 80   # above this → proceed autonomously
  confidence_flag: 60         # below this → pause for human input

validators:
  run_ruff: true
  run_pylint: true
  run_bandit: true
```

---

## API Keys

| Key | Required | Where to get |
|-----|----------|--------------|\
| `GROQ_API_KEY` | Yes | [console.groq.com](https://console.groq.com) — free, no credit card |
| `WANDB_API_KEY` | Optional | [wandb.ai/authorize](https://wandb.ai/authorize) — free |
| `LANGSMITH_API_KEY` | Optional | [smith.langchain.com](https://smith.langchain.com) — free |

---

## Observability

When `WANDB_API_KEY` is set, Forge logs:
- Per-agent latency, token count, and confidence score
- Pipeline-level summary on completion
- Priority weights used for the run

When `LANGSMITH_API_KEY` is set, full LLM traces are recorded for `RequirementAgent`, `PlanningAgent`, `ArchitectureValidatorAgent`, and `DebugAgent` — giving end-to-end visibility into every prompt and completion.

Both are optional. The pipeline runs fully without them.

---

## Output Artifacts

After a pipeline run, the following files are written:

| File | Contents |
|------|----------|
| `outputs/summary.json` | Full pipeline report including decision audit, metrics, file list |
| `outputs/specification.json` | Structured requirement spec produced by RequirementAgent |
| `outputs/security_report.json` | Per-file bandit findings and fix status |
| `outputs/activity.log` | Timestamped agent action log |
| `generated_project/` | Complete project source code |
| `tests/` | TDD-first test suite (written before implementation) |
