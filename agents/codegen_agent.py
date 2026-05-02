"""
agents/codegen_agent.py  —  Forge Engineering Intelligence System

The core problem this file solves:
  An LLM generating code file-by-file will hallucinate imports, redefine shared
  objects (Base, engine, app), use deprecated APIs (Pydantic v1, old SQLAlchemy
  declarative_base), and treat POST bodies as query params — because its training
  data is full of these patterns and it has no explicit contract to follow.

The solution:
  Before generating any file, derive a PER-FILE CONTRACT that specifies:
    - Exact imports the file must use (and where from)
    - Exact classes / functions to define
    - Framework version patterns to follow (Pydantic v2, SQLAlchemy 2.x, etc.)
    - What NOT to define (shared objects defined elsewhere)
  The LLM generates code to satisfy this contract. The contract is also used
  during retry so the fix is targeted at the actual root cause.
"""
from __future__ import annotations
import time
from pathlib import Path

from core.llm import get_client
from core.validators import validate_file, run_pytest, run_syntax_check
from core.observability import log_action, log_agent_metrics, log_metric
from core.state import ForgeState
from agents.debug_agent import run_targeted_fix


# ── Boilerplate file patterns — use fast model, single shot ───────────────────
BOILERPLATE_PATTERNS = (
    "__init__.py", "conftest.py", "constants.py", ".env.example", "setup.py",
)

# ── Framework version patterns injected into every prompt ─────────────────────
FRAMEWORK_PATTERNS = {
    "fastapi": """
FastAPI patterns to follow:
- Import: from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, Query
- Use APIRouter with prefix, not bare @app decorators in router files
- POST/PUT bodies: always use a Pydantic schema parameter (body: SchemaName), NEVER query params
- Dependency injection: def get_db() -> Generator[Session, None, None] in database.py, use Depends(get_db)
- Use lifespan context manager for startup (not @app.on_event)
- Return proper HTTP status codes: 201 for create, 204 for delete, 404 for not found
- Use HTTPException(status_code=404, detail="...") not return {"error": "..."}
""",
    "pydantic_v2": """
Pydantic v2 patterns (REQUIRED — do NOT use v1 patterns):
- Use: from pydantic import BaseModel, field_validator, ConfigDict
- Class config: model_config = ConfigDict(from_attributes=True)  ← NOT class Config: orm_mode = True
- Validators: @field_validator('field_name') @classmethod def name(cls, v): ...
- Do NOT use: @validator, orm_mode, class Config
""",
    "sqlalchemy_2": """
SQLAlchemy 2.x patterns (REQUIRED — do NOT use legacy patterns):
- Base: class Base(DeclarativeBase): pass  ← NOT declarative_base()
- Import: from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
- Engine: from sqlalchemy import create_engine
- Session dependency yields Session, not SessionLocal()
- Models inherit from Base (defined ONCE in database.py — import it, never redefine)
- Use mapped_column() for columns: id: Mapped[int] = mapped_column(primary_key=True)
  OR classic Column() syntax — pick one and be consistent across all model files
""",
    "jwt_auth": """
JWT / OAuth2 authentication patterns (Tier 4):
- Use python-jose for JWT: from jose import JWTError, jwt
- Use passlib for password hashing: from passlib.context import CryptContext
- OAuth2 password flow: from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
- Token creation: jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)
- Token verification: jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
- RBAC: add a `role` field to the User model (enum: admin/user/readonly)
- Protect routes: async def route(current_user: User = Depends(get_current_user))
- get_current_user dependency: decodes JWT, queries DB for user, raises 401 if invalid
- Never store plain-text passwords — always hash with pwd_context.hash()
- Return 401 for invalid credentials, 403 for insufficient permissions
""",
    "mongodb": """
MongoDB with Motor (async) patterns (Tier 5):
- Use motor.motor_asyncio: from motor.motor_asyncio import AsyncIOMotorClient
- Client: client = AsyncIOMotorClient(MONGODB_URL)
- Database: db = client[DATABASE_NAME]
- Collections: collection = db["collection_name"]
- All DB operations are async: await collection.find_one({...})
- ObjectId handling: from bson import ObjectId; str(doc["_id"]) to serialize
- Change Streams: async with collection.watch() as stream: async for change in stream: ...
- For join simulation: use $lookup in aggregation pipelines
- Historical sync: query source, insert_many() into target collection in batches
- Always handle pymongo.errors.PyMongoError in try/except blocks
""",
    "async_http": """
Async HTTP client patterns (Tier 3):
- Use httpx for async requests: import httpx
- Always use async with httpx.AsyncClient() as client: response = await client.get(url)
- Set timeouts: httpx.AsyncClient(timeout=30.0)
- Retry logic: implement exponential backoff with tenacity or manual retry loop
- Handle httpx.TimeoutException, httpx.ConnectError separately from HTTP errors
- Check response.raise_for_status() then parse response.json()
- Use asyncio.gather() for concurrent requests, not sequential awaits
""",
}


# ── System prompts ─────────────────────────────────────────────────────────────

CODEGEN_SYSTEM = """You are an expert Python developer generating production code for a software project.

ABSOLUTE RULES — violating any of these causes the build to fail:
1. Output ONLY raw Python code. Zero markdown. Zero ``` fences. Zero explanation.
2. The very first character must be `"` (docstring) or a letter (import/code).
3. Follow the EXACT import paths specified in the contract — never import from a module not listed.
4. Never redefine objects that the contract says are defined elsewhere (Base, engine, app, router, etc.).
5. POST/PUT endpoints always take a Pydantic schema as the request body, never query params.
6. Use the framework version patterns specified — Pydantic v2, SQLAlchemy 2.x, FastAPI lifespan.
7. Every function must have type hints. No bare `except:`. No unused imports.
8. Generate COMPLETE, RUNNABLE files — never leave TODOs, stubs, or pass statements in real logic.
9. All files in the project are listed in the PROJECT MANIFEST — only import from files listed there.
10. If a module is not in the manifest, do NOT import it (except stdlib and listed third-party packages)."""

RETRY_SYSTEM = """You are fixing a Python file that failed validation.

RULES:
1. Output ONLY the complete corrected file. No markdown. No explanation.
2. Read the error carefully and fix the ROOT CAUSE, not just the symptom.
3. If the error says "Base not defined" — add the correct import from the contract.
4. If the error says "query param" issue — change to a Pydantic body schema.
5. Do not change anything unrelated to the reported error.
6. The file must be COMPLETE — include all original functions plus the fix.
7. Preserve all existing working logic — only fix what the error identifies."""

EXTEND_SYSTEM = """You are extending an existing Python file with new functionality.

RULES:
1. Output the COMPLETE file — all existing code PLUS the new additions.
2. Never remove or modify existing functions/classes.
3. Add new code below the existing code.
4. Match the existing code style exactly.
5. No markdown. No explanation. Raw Python only."""

BOILERPLATE_SYSTEM = "Write a Python file. Output ONLY raw Python code. No markdown fences. No explanation."


# ── Main entry point ──────────────────────────────────────────────────────────

def run(state: ForgeState) -> ForgeState:
    log_action("CodeGenAgent", "Starting code generation pipeline")

    task_plan        = state.get("task_plan", [])
    architecture     = state.get("architecture", {})
    spec             = state.get("structured_spec", {})
    test_files       = state.get("test_files", {})
    generated_files  = state.get("generated_files", {})
    priority_weights = state.get("priority_weights", {})
    debug_attempts   = state.get("debug_attempts", {})
    test_results     = state.get("test_results", {})
    file_diffs       = state.get("file_diffs", [])
    ruff_results     = state.get("ruff_results", {})
    pylint_results   = state.get("pylint_results", {})

    project_dir = Path("./generated_project")
    project_dir.mkdir(exist_ok=True)

    client = get_client()
    tasks_done   = 0
    tasks_failed = 0

    # ── Build project-wide context once ──────────────────────────────────────
    stack        = spec.get("inferred_stack", {})
    fw_patterns  = _detect_framework_patterns(stack, architecture)
    file_registry= {}  # filepath → {role, defines, exports} — built as files generate
    # Project manifest: every file that will exist in this project, used to validate imports
    project_manifest = _build_project_manifest(task_plan)

    for task in task_plan:
        if task["status"] != "pending":
            continue

        task["status"] = "in_progress"
        log_action("CodeGenAgent", f"▶ Task {task['id']}: {task['title']}")

        for filepath in task.get("files", []):

            # ── Skip QAAgent test files ───────────────────────────────────────
            if filepath in test_files:
                log_action("CodeGenAgent", f"  ⏭ {filepath} (QAAgent) — skipping")
                if filepath not in generated_files:
                    _accept_file(filepath, test_files[filepath], task, 0,
                                 generated_files, file_diffs, project_dir)
                continue

            # ── Check for shared file (already generated, needs extension) ───
            existing = generated_files.get(filepath, "")
            is_extension = bool(existing and "PLACEHOLDER" not in existing[:80])

            # ── Build per-file contract ───────────────────────────────────────
            contract = _build_file_contract(
                filepath=filepath,
                task=task,
                architecture=architecture,
                spec=spec,
                generated_files=generated_files,
                file_registry=file_registry,
                fw_patterns=fw_patterns,
                is_extension=is_extension,
                existing_content=existing,
                project_manifest=project_manifest,
            )

            success = _generate_one_file(
                state=state,
                filepath=filepath,
                task=task,
                contract=contract,
                generated_files=generated_files,
                client=client,
                debug_attempts=debug_attempts,
                ruff_results=ruff_results,
                pylint_results=pylint_results,
                file_diffs=file_diffs,
                project_dir=project_dir,
                is_extension=is_extension,
                existing_content=existing,
            )

            if not success:
                log_action("CodeGenAgent", f"  ✗ Failed: {filepath}", level="WARN")
            else:
                # Update file registry AND manifest so later files see real exports
                _register_file(filepath, generated_files.get(filepath, ""),
                               file_registry, project_manifest)

        # ── Run tests for this task ───────────────────────────────────────────
        test_file = task.get("test_file")
        if test_file and test_file in test_files:
            _run_task_tests(task, test_file, test_files, generated_files,
                            test_results, project_dir)

        # ── Task status ───────────────────────────────────────────────────────
        py_files = [f for f in task.get("files", []) if f.endswith(".py")]
        placeholder_count = sum(
            1 for f in py_files
            if "PLACEHOLDER" in (generated_files.get(f) or "")[:80]
        )
        if placeholder_count == 0 and py_files:
            task["status"] = "done"
            tasks_done += 1
        elif placeholder_count < len(py_files):
            task["status"] = "done"
            tasks_done += 1
        else:
            task["status"] = "failed"
            tasks_failed += 1

    metrics = {
        "tasks_completed":       tasks_done,
        "tasks_failed":          tasks_failed,
        "files_generated":       len(generated_files),
        "first_attempt_pass_rate": _calc_first_attempt_rate(debug_attempts, generated_files),
    }
    log_agent_metrics("CodeGenAgent", metrics)
    log_metric("codegen/files_generated", len(generated_files))
    log_metric("codegen/first_attempt_pass_rate", metrics["first_attempt_pass_rate"])

    state["generated_files"]              = generated_files
    state["debug_attempts"]               = debug_attempts
    state["test_results"]                 = test_results
    state["ruff_results"]                 = ruff_results
    state["pylint_results"]               = pylint_results
    state["file_diffs"]                   = file_diffs
    state["task_plan"]                    = task_plan
    state["agent_metrics"]["CodeGenAgent"] = metrics
    state["phase"]                        = "codegen_done"

    log_action("CodeGenAgent",
               f"Code generation complete — {tasks_done} tasks done, {tasks_failed} failed")
    return state


# ── Core generation ───────────────────────────────────────────────────────────

def _generate_one_file(
    state, filepath, task, contract,
    generated_files, client, debug_attempts,
    ruff_results, pylint_results, file_diffs, project_dir,
    max_retries=3,
    is_extension=False,
    existing_content="",
) -> bool:
    is_python = filepath.endswith(".py")

    if not is_python:
        return _generate_non_python(state, filepath, contract, client,
                                    generated_files, project_dir)

    if _is_boilerplate(filepath):
        return _generate_boilerplate(state, filepath, contract, client,
                                     generated_files, project_dir,
                                     debug_attempts, file_diffs)

    system = EXTEND_SYSTEM if is_extension else CODEGEN_SYSTEM
    first_user = _build_user_prompt(filepath, task, contract,
                                    existing_content if is_extension else "")

    attempt   = 0
    last_error = None
    last_code  = ""

    while attempt < max_retries:
        attempt += 1
        debug_attempts[filepath] = attempt

        if attempt == 1:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": first_user},
            ]
            response, usage = client.call(messages, agent_name="CodeGenAgent")
            _track(state, usage)
            code = _clean_code(response)
        else:
            # Retry: hand off to DebugAgent which classifies the error type
            # (syntax / import / pydantic / fastapi / logic / lint) and applies
            # a targeted strategy — far more reliable than a generic "fix it" prompt.
            log_action("CodeGenAgent",
                       f"  → Handing off to DebugAgent (attempt {attempt}): {filepath}",
                       f"error_type will be classified from: {last_error[:80]}")
            contract_summary = _contract_to_text(contract)
            fixed_code, dbg_usage = run_targeted_fix(
                code=last_code,
                error_info=last_error,
                filepath=filepath,
                contract={
                    "import_rules": contract.get("import_rules", ""),
                    "must_not_define": contract.get("must_not_define", ""),
                    "framework_patterns": contract.get("framework_patterns", ""),
                    "contract_summary": contract_summary,
                },
                agent_name="DebugAgent",
            )
            # Track the debug agent's token usage against the pipeline totals
            _track(state, dbg_usage)
            code = fixed_code

        if len(code.strip()) < 30:
            last_error = "LLM returned empty or trivially short response"
            last_code  = code
            continue

        # Run a quick syntax check first — catches structural issues immediately
        syntax_check = run_syntax_check(code)
        if not syntax_check["passed"]:
            last_error = f"SyntaxError: {syntax_check.get('error', 'unknown syntax error')}"
            last_code  = code
            log_action("CodeGenAgent",
                       f"  attempt {attempt}/{max_retries} syntax error: {filepath}",
                       last_error[:120], "WARN")
            continue

        validation = validate_file(code, filepath)
        ruff_results[filepath]   = validation.get("ruff")
        pylint_results[filepath] = validation.get("pylint")

        if validation["overall_passed"]:
            _accept_file(filepath, code, task, attempt,
                         generated_files, file_diffs, project_dir)
            log_action("CodeGenAgent", f"  ✓ {filepath}",
                       f"attempt={attempt} lines={len(code.splitlines())}")
            return True
        else:
            last_error = validation.get("blocking_error") or "Validation failed"
            last_code  = code
            log_action("CodeGenAgent",
                       f"  attempt {attempt}/{max_retries} failed: {filepath}",
                       last_error[:120], "WARN")

    # Accept with lint warnings if syntax is clean (lint issues are non-blocking)
    if last_code and run_syntax_check(last_code)["passed"]:
        log_action("CodeGenAgent", f"  ⚠ Accepting with warnings: {filepath}", level="WARN")
        _accept_file(filepath, last_code, task, attempt,
                     generated_files, file_diffs, project_dir)
        return True

    placeholder = _make_placeholder(filepath, task, last_error)
    _accept_file(filepath, placeholder, task, attempt,
                 generated_files, file_diffs, project_dir, status="placeholder")
    log_action("CodeGenAgent", f"  ✗ PLACEHOLDER: {filepath}", last_error[:120], "ERROR")
    return False


def _generate_boilerplate(
    state, filepath, contract, client,
    generated_files, project_dir, debug_attempts, file_diffs,
) -> bool:
    task_stub = {"id": "boilerplate", "title": filepath, "task_id": "bp"}
    prompt = (
        f"Write `{filepath}`.\n"
        f"Purpose: {contract.get('purpose', filepath)}\n"
        f"Required content:\n{contract.get('required_content', 'Standard Python module')}\n\n"
        "Output ONLY raw Python code."
    )
    response, usage = client.call_fast(
        [{"role": "system", "content": BOILERPLATE_SYSTEM},
         {"role": "user",   "content": prompt}],
        agent_name="CodeGenAgent",
    )
    _track(state, usage)
    code = _clean_code(response) or f'"""{filepath}"""\n'
    debug_attempts[filepath] = 1
    _accept_file(filepath, code, task_stub, 1, generated_files, file_diffs, project_dir)
    log_action("CodeGenAgent", f"  ✓ {filepath} (boilerplate)")
    return True


def _generate_non_python(
    state, filepath, contract, client, generated_files, project_dir,
) -> bool:
    prompt = (
        f"Generate `{filepath}`.\n"
        f"Context: {contract.get('purpose', '')}\n"
        f"Details: {contract.get('required_content', '')}\n"
        "Output ONLY the file content. No explanation."
    )
    response, usage = client.call_fast(
        [{"role": "user", "content": prompt}], agent_name="CodeGenAgent"
    )
    _track(state, usage)
    content = _clean_code(response)
    generated_files[filepath] = content
    out = project_dir / filepath
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    log_action("CodeGenAgent", f"  ✓ {filepath} (non-Python)")
    return True


# ── Contract system ───────────────────────────────────────────────────────────

def _build_file_contract(
    filepath: str,
    task: dict,
    architecture: dict,
    spec: dict,
    generated_files: dict,
    file_registry: dict,
    fw_patterns: str,
    is_extension: bool,
    existing_content: str,
    project_manifest: dict = None,
) -> dict:
    """
    Build a per-file contract: what the file must import, define, and implement.
    This is the key differentiator — each file gets a precise specification
    derived from the actual project architecture, not a generic description.
    """
    name       = Path(filepath).stem.lower()
    parent_dir = Path(filepath).parent.as_posix()
    all_files  = list(generated_files.keys()) + [filepath]

    # ── Identify role of this file ────────────────────────────────────────────
    role = _classify_file_role(filepath, task)

    # ── Build import map: what exists and where ───────────────────────────────
    import_map = _build_import_map(filepath, all_files, file_registry, generated_files)

    # ── Extract endpoints, models, schemas from architecture ──────────────────
    endpoints = [ep for ep in architecture.get("api_endpoints", [])
                 if _endpoint_belongs_to_file(ep, filepath, task)]
    models    = architecture.get("data_models", [])
    schemas   = _derive_schemas_for_file(filepath, role, models, task)

    # ── Build the contract dict ───────────────────────────────────────────────
    contract = {
        "filepath":        filepath,
        "role":            role,
        "purpose":         task.get("description", task.get("title", "")),
        "framework_patterns": fw_patterns,
        "import_rules":    import_map,
        "must_define":     _what_to_define(role, filepath, endpoints, models, schemas, task),
        "must_not_define": _what_not_to_define(role, file_registry),
        "endpoints":       endpoints,
        "models":          models[:3],
        "schemas":         schemas,
        "is_extension":    is_extension,
        "required_content": _required_content_hint(role, task, architecture, spec),
        "project_manifest": project_manifest or {},
    }

    return contract


def _build_user_prompt(filepath: str, task: dict, contract: dict, existing: str) -> str:
    """
    Build the user-facing prompt from the contract.

    ORDERING MATTERS: Put the most critical context FIRST because:
    1. If the prompt is long, LLMs attend more to early content
    2. The token trimmer preserves the head+tail, so put contract critical
       sections at the top and the less critical descriptive sections lower
    """
    parts = []

    # ── 1. FILE IDENTITY & ROLE (always first) ────────────────────────────────
    if existing:
        parts.append(
            f"EXTEND this existing file: `{filepath}`\n"
            f"Keep ALL existing code. Add the new functionality below.\n"
            f"Role: {contract['role']} | Task: {task['title']}"
        )
    else:
        parts.append(
            f"Write file: `{filepath}`\n"
            f"Role: {contract['role']} | Task: {task['title']}"
        )

    # ── 2. IMPORT RULES (critical — must come early) ──────────────────────────
    if contract.get("import_rules"):
        parts.append(f"\nIMPORT RULES (follow exactly — only import from these paths):\n{contract['import_rules']}")

    # ── 3. MUST NOT DEFINE (prevents hallucination of shared objects) ─────────
    if contract.get("must_not_define"):
        parts.append(f"\nDO NOT DEFINE (import these — they exist in other files):\n{contract['must_not_define']}")

    # ── 4. PROJECT MANIFEST (what files exist — prevents invalid imports) ─────
    manifest = contract.get("project_manifest", {})
    if manifest:
        manifest_lines = []
        for fp, info in list(manifest.items())[:20]:  # cap at 20 files
            mod = fp.replace("/", ".").removesuffix(".py")
            exports = info.get("exports", [])
            if exports:
                manifest_lines.append(f"  {mod}: exports {', '.join(exports[:5])}")
            else:
                manifest_lines.append(f"  {mod}")
        parts.append(f"\nPROJECT MANIFEST (valid import targets):\n" + "\n".join(manifest_lines))

    # ── 5. FRAMEWORK PATTERNS ─────────────────────────────────────────────────
    if contract.get("framework_patterns"):
        parts.append(f"\n{contract['framework_patterns']}")

    # ── 6. WHAT TO DEFINE ────────────────────────────────────────────────────
    if contract.get("must_define"):
        parts.append(f"\nMUST DEFINE:\n{contract['must_define']}")

    # ── 7. ENDPOINTS (if this is a router/main file) ──────────────────────────
    if contract.get("endpoints"):
        ep_lines = []
        for ep in contract["endpoints"]:
            ep_lines.append(
                f"  {ep.get('method','GET')} {ep.get('path','/')} — {ep.get('description','')}"
            )
        parts.append(f"\nENDPOINTS TO IMPLEMENT:\n" + "\n".join(ep_lines))

    # ── 8. SCHEMAS (if this is a schema file) ────────────────────────────────
    if contract.get("schemas"):
        parts.append(f"\nSCHEMAS:\n{contract['schemas']}")

    # ── 9. DATA MODELS (if this is a model/database file) ────────────────────
    if contract.get("models") and contract["role"] in ("model", "database"):
        model_lines = []
        for m in contract["models"]:
            fields = ", ".join(
                f"{f['name']}: {f['type']}"
                for f in m.get("fields", [])[:8]
            )
            model_lines.append(f"  {m.get('name','')}: {fields}")
        parts.append(f"\nDATA MODELS:\n" + "\n".join(model_lines))

    # ── 10. TASK DESCRIPTION ─────────────────────────────────────────────────
    desc = task.get("description", "")
    if desc:
        parts.append(f"\nTask description: {desc[:500]}")

    # ── 11. EXISTING FILE (for extensions — comes late so contract rules win) ──
    if existing:
        # Show the full existing file so the LLM can preserve all of it
        parts.append(f"\nExisting file:\n{existing}")
        parts.append(f"\nNew task to add: {task['title']}\n{task.get('description', '')[:400]}")

    # ── 12. FINAL INSTRUCTION (always last) ───────────────────────────────────
    parts.append(
        f"\nOutput ONLY raw Python code for `{filepath}`. "
        "No markdown. No ``` fences. No explanation. "
        "The FIRST character must be '\"' or a letter. "
        "Generate the COMPLETE file — no TODOs, no stubs."
    )

    return "\n".join(parts)


def _contract_to_text(contract: dict) -> str:
    """Compact text version of the contract for use in retry prompts."""
    lines = [
        f"Role: {contract.get('role', 'unknown')}",
        f"Purpose: {contract.get('purpose', '')[:200]}",
    ]
    if contract.get("import_rules"):
        lines.append(f"Import rules:\n{contract['import_rules']}")
    if contract.get("must_not_define"):
        lines.append(f"Do NOT define:\n{contract['must_not_define']}")
    if contract.get("framework_patterns"):
        lines.append(contract["framework_patterns"][:400])
    return "\n".join(lines)


# ── Contract derivation helpers ────────────────────────────────────────────────

def _classify_file_role(filepath: str, task: dict) -> str:
    name  = Path(filepath).stem.lower()
    parts = filepath.lower().split("/")
    title = task.get("title", "").lower()

    if "database" in name or "db" in name:          return "database"
    if "model"    in name:                           return "model"
    if "schema"   in name:                           return "schema"
    if "router"   in parts or "router" in name:      return "router"
    if "route"    in name or "endpoint" in name:     return "router"
    if "main"     in name:                           return "main"
    if "middleware" in name:                          return "middleware"
    if "auth"     in name or "security" in name:     return "auth"
    if "config"   in name or "settings" in name:     return "config"
    if "test"     in name or "test" in parts:        return "test"
    if "crud"     in name:                           return "crud"
    return "module"


def _build_import_map(
    filepath: str,
    all_files: list,
    file_registry: dict,
    generated_files: dict,
) -> str:
    """
    Build explicit import rules for this file based on what's been generated.
    Uses REAL export names from file_registry — no more <ModelClassName> placeholders.
    This is what prevents the LLM from redefining Base, engine, app etc.
    """
    rules = []
    role  = _classify_file_role(filepath, {"title": ""})

    def to_module(fp: str) -> str:
        return fp.replace("/", ".").removesuffix(".py")

    def get_real_classes(fp: str, filter_fn=None) -> list[str]:
        """Get real class names from file_registry, or parse from generated code."""
        if fp in file_registry:
            classes = file_registry[fp].get("classes", [])
            return [c for c in classes if (filter_fn is None or filter_fn(c))]
        # Fall back to parsing the generated code directly
        code = generated_files.get(fp, "")
        return [
            line.split("(")[0].replace("class ", "").strip()
            for line in code.splitlines()
            if line.startswith("class ") and (filter_fn is None or True)
        ][:6]

    def get_real_functions(fp: str) -> list[str]:
        if fp in file_registry:
            return file_registry[fp].get("functions", [])[:6]
        code = generated_files.get(fp, "")
        return [
            line.split("(")[0].replace("def ", "").replace("async def ", "").strip()
            for line in code.splitlines()
            if (line.startswith("def ") or line.startswith("async def "))
            and not line.strip().startswith("_")
        ][:6]

    # ── Find key files ────────────────────────────────────────────────────────
    db_file = next(
        (f for f in all_files if "database" in Path(f).stem.lower() and f.endswith(".py")), None
    )
    model_file = next(
        (f for f in all_files
         if "model" in Path(f).stem.lower() and f.endswith(".py") and f != filepath), None
    )
    schema_file = next(
        (f for f in all_files
         if "schema" in Path(f).stem.lower() and f.endswith(".py") and f != filepath), None
    )
    auth_file = next(
        (f for f in all_files
         if ("auth" in Path(f).stem.lower() or "security" in Path(f).stem.lower())
         and f.endswith(".py") and f != filepath), None
    )
    config_file = next(
        (f for f in all_files
         if ("config" in Path(f).stem.lower() or "settings" in Path(f).stem.lower())
         and f.endswith(".py") and f != filepath), None
    )

    # ── database.py imports ───────────────────────────────────────────────────
    if db_file and filepath != db_file:
        db_mod = to_module(db_file)
        db_fns = get_real_functions(db_file)
        has_get_db = "get_db" in db_fns or not db_fns  # assume get_db exists if unknown

        if role == "model":
            rules.append(f"from {db_mod} import Base  # Base defined ONCE here — never redefine")
        elif role in ("router", "crud"):
            imports = ["get_db"]
            if has_get_db or not db_fns:
                rules.append(f"from {db_mod} import {', '.join(imports)}")
        elif role == "main":
            rules.append(f"from {db_mod} import Base, engine, init_db  # for startup")

    # ── models imports ────────────────────────────────────────────────────────
    if model_file and filepath != model_file and role in ("router", "crud", "main", "schema"):
        mod_mod = to_module(model_file)
        classes = get_real_classes(model_file, lambda c: not c.startswith("Base"))
        if classes:
            rules.append(f"from {mod_mod} import {', '.join(classes[:4])}")
        else:
            # Model file not generated yet — provide a descriptive placeholder
            rules.append(f"from {mod_mod} import <ModelClass>  # replace with actual model name")

    # ── schemas imports ───────────────────────────────────────────────────────
    if schema_file and filepath != schema_file and role in ("router", "crud", "main"):
        sch_mod = to_module(schema_file)
        classes = get_real_classes(schema_file, lambda c: "Base" not in c)
        if classes:
            rules.append(f"from {sch_mod} import {', '.join(classes[:6])}")
        else:
            rules.append(f"from {sch_mod} import <SchemaCreate, SchemaUpdate, SchemaResponse>")

    # ── auth/security imports ─────────────────────────────────────────────────
    if auth_file and filepath != auth_file and role in ("router", "main", "middleware"):
        auth_mod = to_module(auth_file)
        auth_fns = get_real_functions(auth_file)
        if auth_fns:
            rules.append(f"from {auth_mod} import {', '.join(auth_fns[:4])}")
        else:
            rules.append(f"from {auth_mod} import get_current_user  # auth dependency")

    # ── config/settings imports ───────────────────────────────────────────────
    if config_file and filepath != config_file and role in ("database", "main", "auth"):
        cfg_mod = to_module(config_file)
        cfg_classes = get_real_classes(config_file)
        if cfg_classes:
            rules.append(f"from {cfg_mod} import {cfg_classes[0]}")
        else:
            rules.append(f"from {cfg_mod} import settings  # app configuration")

    return "\n".join(rules) if rules else ""


def _what_to_define(
    role: str, filepath: str, endpoints: list, models: list, schemas: str, task: dict
) -> str:
    lines = []
    if role == "database":
        lines += [
            "class Base(DeclarativeBase): pass  ← the ONLY place Base is defined",
            "engine = create_engine(DATABASE_URL, ...)",
            "SessionLocal = sessionmaker(...)",
            "def get_db() -> Generator[Session, None, None]  ← yields session",
            "def init_db() -> None  ← calls Base.metadata.create_all()",
        ]
    elif role == "model":
        for m in models[:3]:
            fields = ", ".join(f"{f['name']}: {f['type']}" for f in m.get("fields", [])[:6])
            lines.append(f"class {m.get('name','')}(Base): {fields}")
    elif role == "schema":
        lines.append(schemas or "Pydantic schemas with field_validator, ConfigDict(from_attributes=True)")
    elif role == "router":
        for ep in endpoints[:8]:
            lines.append(f"{ep.get('method','GET')} {ep.get('path','/')} → {ep.get('description','')}")
    elif role == "main":
        lines += [
            "app = FastAPI(title=..., lifespan=lifespan)",
            "@asynccontextmanager async def lifespan(app): init_db(); yield",
            "app.include_router(router, prefix=...)",
            "GET /health → {status: ok}",
        ]
    elif role == "middleware":
        lines.append("Middleware classes and validation helper functions")

    return "\n".join(f"  - {l}" for l in lines) if lines else task.get("description", "")[:300]


def _what_not_to_define(role: str, file_registry: dict) -> str:
    """List shared objects that must be imported, not redefined."""
    dont = []
    if role != "database":
        dont.append("Base, engine, SessionLocal, get_db  ← defined in database.py, import them")
    if role not in ("main",):
        dont.append("app = FastAPI()  ← defined in main.py only")
    if role not in ("router", "main"):
        dont.append("APIRouter  ← only create router in router files, not in main or models")
    return "\n".join(f"  - {d}" for d in dont) if dont else ""


def _derive_schemas_for_file(filepath: str, role: str, models: list, task: dict) -> str:
    if role != "schema":
        return ""
    lines = []
    for m in models[:3]:
        name  = m.get("name", "Item")
        fields = m.get("fields", [])
        field_str = "\n    ".join(
            f"{f['name']}: {_py_type(f.get('type', 'str'))}"
            for f in fields if f.get("name") != "id"
        )
        update_fields = "\n    ".join(
            f"{f['name']}: Optional[{_py_type(f.get('type', 'str'))}] = None"
            for f in fields if f.get("name") != "id"
        ) or "pass"
        lines.append(
            f"class {name}Create(BaseModel):\n    {field_str or 'pass'}\n\n"
            f"class {name}Update(BaseModel):  # all fields Optional\n"
            f"    {update_fields}\n\n"
            f"class {name}Response({name}Create):\n"
            f"    id: int\n    model_config = ConfigDict(from_attributes=True)"
        )
    return "\n\n".join(lines)


def _endpoint_belongs_to_file(ep: dict, filepath: str, task: dict) -> bool:
    """Check if an endpoint should be implemented in this file."""
    name  = Path(filepath).stem.lower()
    path  = ep.get("path", "").lower()
    desc  = ep.get("description", "").lower()
    title = task.get("title", "").lower()
    # Endpoint belongs if the resource name matches the file name
    # or the task title mentions the endpoint method/path
    resource = path.split("/")[1] if "/" in path else ""
    return (resource in name or name in resource or
            ep.get("method", "").lower() in title or
            any(word in title for word in path.split("/")))


def _required_content_hint(role: str, task: dict, architecture: dict, spec: dict) -> str:
    """One-line hint used for boilerplate and non-Python file generation."""
    if role == "config":
        return "Settings class using pydantic-settings or os.environ"
    desc = task.get("description", task.get("title", ""))
    stack = spec.get("inferred_stack", {})
    return f"{desc} — Stack: {', '.join(f'{k}={v}' for k,v in stack.items())}"


def _detect_framework_patterns(stack: dict, architecture: dict) -> str:
    """Build the framework version pattern string for injection into every prompt."""
    stack_str = str(stack).lower()
    arch_str  = str(architecture).lower()
    patterns  = []

    if "fastapi" in stack_str or any(
        "fastapi" in str(d).lower()
        for d in architecture.get("key_decisions", [])
    ):
        patterns.append(FRAMEWORK_PATTERNS["fastapi"])

    # Always inject Pydantic v2 — it's the current standard
    patterns.append(FRAMEWORK_PATTERNS["pydantic_v2"])

    # SQLAlchemy 2.x — inject if any relational DB is mentioned
    if any(x in stack_str for x in ("sqlite", "postgres", "mysql", "sql")) and \
       "mongo" not in stack_str:
        patterns.append(FRAMEWORK_PATTERNS["sqlalchemy_2"])

    # Tier 4: JWT / OAuth2 auth
    if any(x in stack_str or x in arch_str
           for x in ("jwt", "oauth", "auth", "rbac", "bearer", "token")):
        patterns.append(FRAMEWORK_PATTERNS["jwt_auth"])

    # Tier 5: MongoDB
    if any(x in stack_str or x in arch_str
           for x in ("mongo", "motor", "change stream", "bson")):
        patterns.append(FRAMEWORK_PATTERNS["mongodb"])

    # Tier 3: Async HTTP integrations
    if any(x in stack_str or x in arch_str
           for x in ("httpx", "aiohttp", "async.*http", "third.party", "external api")):
        patterns.append(FRAMEWORK_PATTERNS["async_http"])

    return "\n".join(patterns)


def _py_type(schema_type: str) -> str:
    """Convert architecture schema type to Python type hint."""
    mapping = {
        "int": "int", "integer": "int", "float": "float", "number": "float",
        "str": "str", "string": "str", "bool": "bool", "boolean": "bool",
        "date": "date", "datetime": "datetime", "list": "list", "dict": "dict",
    }
    return mapping.get(schema_type.lower(), "str")


# ── File registry and manifest ────────────────────────────────────────────────

def _build_project_manifest(task_plan: list) -> dict:
    """
    Build the full map of every file the project will contain.
    Populated with exports as each file is generated via _register_file.
    Gives every codegen prompt a concrete list of valid import targets.
    """
    manifest = {}
    for task in task_plan:
        for fp in task.get("files", []):
            if fp not in manifest:
                manifest[fp] = {
                    "role": _classify_file_role(fp, task),
                    "exports": [],   # filled in as files are generated
                }
    return manifest


def _register_file(filepath: str, code: str, file_registry: dict,
                   project_manifest: dict = None) -> None:
    """
    Track what classes and functions a file exports.
    Also updates project_manifest so later prompts see real export names.
    """
    classes   = []
    functions = []
    for line in (code or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("class ") and "(" in stripped:
            cls_name = stripped.split("(")[0].replace("class ", "").strip()
            if cls_name and cls_name[0].isupper():
                classes.append(cls_name)
        elif stripped.startswith("def ") or stripped.startswith("async def "):
            fn = stripped.split("(")[0].replace("async def ", "").replace("def ", "").strip()
            if fn and not fn.startswith("_"):
                functions.append(fn)

    exports = classes + functions
    file_registry[filepath] = {"classes": classes, "functions": functions}

    if project_manifest is not None and filepath in project_manifest:
        project_manifest[filepath]["exports"] = exports[:10]


# ── Test runner ───────────────────────────────────────────────────────────────

def _run_task_tests(
    task, test_file, test_files, generated_files, test_results, project_dir
) -> None:
    test_code  = test_files.get(test_file, "")
    impl_files = [f for f in task.get("files", []) if f.endswith(".py")]
    if not impl_files:
        return
    impl_filepath = impl_files[0]
    impl_code     = generated_files.get(impl_filepath, "")
    if not impl_code or "PLACEHOLDER" in impl_code[:80]:
        return
    result = run_pytest(
        test_code, impl_code, Path(impl_filepath).name,
        all_generated_files=generated_files,
    )
    test_results[test_file] = result
    status = "PASS" if result["passed"] else "FAIL"
    log_action("CodeGenAgent", f"  Tests {status}: {test_file}",
               f"passed={result['tests_passed']} failed={result['tests_failed']}")


# ── Utility helpers ───────────────────────────────────────────────────────────

def _accept_file(
    filepath, code, task, attempt, generated_files, file_diffs, project_dir,
    status="generated",
) -> None:
    generated_files[filepath] = code
    file_diffs.append({
        "filepath": filepath,
        "task_id":  task.get("id", task.get("task_id", "?")),
        "attempt":  attempt,
        "lines":    len(code.splitlines()),
        "status":   status,
    })
    out_path = project_dir / filepath
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code)


def _track(state, usage) -> None:
    state["api_call_count"] = state.get("api_call_count", 0) + 1
    state["total_tokens"]   = state.get("total_tokens", 0) + usage["total_tokens"]


def _is_boilerplate(filepath: str) -> bool:
    return Path(filepath).name in BOILERPLATE_PATTERNS


def _clean_code(raw: str) -> str:
    """Strip markdown fences from LLM output."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        end   = len(lines) - 1
        while end > 0 and lines[end].strip().startswith("```"):
            end -= 1
        raw = "\n".join(lines[1:end + 1])
    return raw.strip()


def _priority_hint(weights: dict) -> str:
    if not weights:
        return "balanced"
    top = max(weights, key=weights.get)
    return {
        "security":      "security-first",
        "quality":       "quality-first",
        "test_coverage": "testability-first",
        "speed":         "speed-first",
        "simplicity":    "simplicity-first",
    }.get(top, "balanced")


def _make_placeholder(filepath: str, task: dict, error: str) -> str:
    return (
        f'"""PLACEHOLDER — {filepath}\nTask: {task.get("title","")}\n'
        f'Generation failed. Error: {(error or "unknown")[:200]}\n"""\n\n'
        f"raise NotImplementedError('Forge placeholder — manual implementation required')\n"
    )


def _calc_first_attempt_rate(debug_attempts: dict, generated_files: dict) -> float:
    if not debug_attempts:
        return 0.0
    first_attempt = sum(1 for v in debug_attempts.values() if v == 1)
    return round(first_attempt / len(debug_attempts) * 100, 1)