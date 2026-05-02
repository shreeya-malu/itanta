"""
agents/debug_agent.py
Classifies failure type and applies targeted, error-type-specific fixes.
Each error class gets a dedicated strategy — generic "fix it" prompts produce
partial fixes that break other parts of the file.
"""
from __future__ import annotations
import time
from core.llm import get_client
from core.observability import log_action


# ── Base system prompt ────────────────────────────────────────────────────────
BASE_SYSTEM = """You are a Python debugging expert. Fix the broken file.

ABSOLUTE RULES:
1. Output ONLY the complete corrected file. No markdown. No ``` fences. No explanation.
2. The FIRST character must be a letter or '"' — not a backtick or space.
3. The fix must be COMPLETE — include ALL original code, not just the fixed section.
4. Fix the ROOT CAUSE of the error, not just suppress it.
5. Do not introduce new imports that weren't in the original unless fixing an ImportError."""


# ── Error-type-specific fix strategies ───────────────────────────────────────
FIX_STRATEGIES = {
    "syntax": """
SYNTAX ERROR FIX STRATEGY:
- Find the exact line mentioned in the error and fix it.
- Common causes: unclosed parenthesis/bracket, missing colon after def/class/if,
  bad indentation, f-string syntax error, unterminated string.
- Check the line BEFORE the reported line — Python often reports the line after the real error.
- Fix indentation by matching surrounding code exactly (4 spaces, no tabs).
""",
    "import": """
IMPORT ERROR FIX STRATEGY:
- If "cannot import name X from Y": the name X doesn't exist in module Y.
  Solution: either change the import to something that DOES exist, or remove it.
- If "No module named X": add X to requirements.txt OR use the correct stdlib alternative.
- Never invent module paths. If unsure, use stdlib equivalents.
- For SQLAlchemy imports: use `from sqlalchemy.orm import DeclarativeBase, Session`
  NOT `from sqlalchemy.ext.declarative import declarative_base` (deprecated).
- For FastAPI: `from fastapi import FastAPI, APIRouter, Depends, HTTPException, status`
""",
    "undefined_name": """
UNDEFINED NAME FIX STRATEGY:
- The variable/class/function is used but never defined or imported.
- If it's a shared object (Base, engine, SessionLocal, get_db, app):
  ADD the correct import at the top of the file.
  Common fixes:
    - Base not defined → from app.database import Base
    - Session not defined → from sqlalchemy.orm import Session
    - Generator not defined → from typing import Generator
    - Optional not defined → from typing import Optional
    - datetime not defined → from datetime import datetime
- If it's a local variable used before assignment: move assignment before use.
- Never remove the usage — fix the missing definition/import.
""",
    "type": """
TYPE / ATTRIBUTE ERROR FIX STRATEGY:
- AttributeError 'X has no attribute Y': check the object type.
  If it's a SQLAlchemy model, the field might be named differently.
  If it's a Pydantic model, use .model_dump() not .dict() (Pydantic v2).
- TypeError 'X() takes N positional arguments': check function signature vs call site.
- For Pydantic v2: replace .dict() → .model_dump(), .from_orm() → model_validate().
- For SQLAlchemy 2.x: Session.execute() returns Row objects, not model objects directly.
  Use session.scalars(select(Model)).all() for model objects.
""",
    "pydantic": """
PYDANTIC v2 FIX STRATEGY:
- Replace ALL v1 patterns with v2:
  - class Config: orm_mode = True  →  model_config = ConfigDict(from_attributes=True)
  - @validator  →  @field_validator (add @classmethod)
  - .dict()     →  .model_dump()
  - .from_orm() →  ModelClass.model_validate(obj)
- Import: from pydantic import BaseModel, ConfigDict, field_validator
- All field_validators need @classmethod decorator.
""",
    "fastapi": """
FASTAPI ENDPOINT FIX STRATEGY:
- POST/PUT body params must be a Pydantic model, NOT query params.
  Wrong: async def create(name: str, price: float)
  Right: async def create(body: ItemCreate, db: Session = Depends(get_db))
- Router files must use APIRouter, not @app.
  Wrong: @app.get("/items")
  Right: router = APIRouter(); @router.get("/items")
- Return HTTPException not plain dicts for errors:
  Wrong: return {"error": "not found"}
  Right: raise HTTPException(status_code=404, detail="Not found")
- Dependency injection: db: Session = Depends(get_db) — not db = SessionLocal()
""",
    "logic": """
LOGIC / TEST FAILURE FIX STRATEGY:
- Read the assertion error carefully — it shows expected vs actual values.
- Check the function return value matches the expected schema.
- For 422 errors: the request body doesn't match the Pydantic schema.
  Check field names, types, and required vs optional.
- For 404 when item should exist: check the DB query or ID lookup logic.
- For 500 errors: add proper exception handling and check for None before attribute access.
""",
    "lint": """
LINT FIX STRATEGY:
- F811 (redefinition): remove the duplicate definition, keep the correct one.
- F401 (unused import): remove the import if truly unused, or add a usage.
- E711 (comparison to None): change `== None` to `is None`.
- E712 (comparison to True/False): change `== True` to just the expression.
- W0611 (unused import): remove it.
- Fix ONLY the specific lint codes reported. Do not refactor unrelated code.
""",
}


def run_targeted_fix(
    code: str,
    error_info: str,
    filepath: str,
    contract: dict = None,
    agent_name: str = "DebugAgent",
) -> tuple[str, dict]:
    """
    Apply a targeted fix. Returns (fixed_code, usage_dict).
    Uses error-type-specific strategy prompts for higher fix accuracy.
    """
    t0 = time.time()
    client = get_client()
    error_type = _classify_error(error_info)
    strategy   = FIX_STRATEGIES.get(error_type, FIX_STRATEGIES["lint"])

    # Build a compact contract summary for grounding
    contract_hint = ""
    if contract:
        import_rules   = contract.get("import_rules", "")
        must_not_define = contract.get("must_not_define", "")
        if import_rules:
            contract_hint += f"\nCORRECT IMPORTS FOR THIS FILE:\n{import_rules}\n"
        if must_not_define:
            contract_hint += f"\nDO NOT DEFINE (import from other files):\n{must_not_define}\n"

    system = BASE_SYSTEM + "\n" + strategy

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"File to fix: `{filepath}`\n"
                f"Error type: {error_type}\n"
                f"Error message:\n{error_info[:800]}\n"
                f"{contract_hint}\n"
                f"Complete broken file:\n{code}\n\n"
                "Output the COMPLETE corrected file. "
                "Include ALL original functions — not just the fixed part. "
                "No markdown. Raw Python only."
            ),
        },
    ]

    response, usage = client.call(messages, agent_name=agent_name)
    fixed = _clean_code(response)
    usage["latency_s"]  = round(time.time() - t0, 3)
    usage["error_type"] = error_type

    log_action("DebugAgent", f"Fix applied to {filepath}",
               f"error_type={error_type} lines={len(fixed.splitlines())}")
    return fixed, usage


def _classify_error(error_info: str) -> str:
    """Map error message to a fix strategy key."""
    s = error_info.lower()

    if "syntaxerror" in s or "indentationerror" in s or "unexpected indent" in s:
        return "syntax"

    if ("cannot import name" in s or "importerror" in s
            or "modulenotfounderror" in s or "no module named" in s):
        return "import"

    if ("nameerror" in s or "is not defined" in s or "undefined" in s):
        return "undefined_name"

    if ("typeerror" in s or "attributeerror" in s
            or "has no attribute" in s or "takes" in s):
        return "type"

    if ("configdict" in s or "orm_mode" in s or "field_validator" in s
            or "pydantic" in s or "basemodel" in s):
        return "pydantic"

    if ("422" in s or "query param" in s or "request body" in s
            or "apiRouter" in s.lower() or "depends" in s):
        return "fastapi"

    if "assertionerror" in s or "assert" in s or "failed" in s:
        return "logic"

    return "lint"


def _clean_code(raw: str) -> str:
    """Strip markdown fences — LLMs sometimes add them despite instructions."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        end = len(lines) - 1
        while end > 0 and lines[end].strip().startswith("```"):
            end -= 1
        raw = "\n".join(lines[1:end + 1])
    return raw.strip()
