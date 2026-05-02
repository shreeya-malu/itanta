"""
core/validators.py
Objective validation tools: ruff, pylint, bandit, pytest.
Each returns a structured result dict for agent consumption.
"""
from __future__ import annotations
import subprocess, json, tempfile, os, sys
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], cwd: str = None, timeout: int = 60) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "TIMEOUT"
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"


def write_temp_file(code: str, suffix: str = ".py") -> str:
    """Write code to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    tmp.write(code)
    tmp.close()
    return tmp.name


def run_ruff(code: str, filepath: str = "code.py") -> dict:
    """Run ruff linter on a code string. Returns structured result."""
    tmp = write_temp_file(code)
    try:
        rc, stdout, stderr = _run(
            ["python", "-m", "ruff", "check", "--output-format=json", tmp]
        )
        issues = []
        try:
            raw = json.loads(stdout) if stdout.strip() else []
            for item in raw:
                issues.append({
                    "code": item.get("code", ""),
                    "message": item.get("message", ""),
                    "line": item.get("location", {}).get("row", 0),
                    "fixable": item.get("fix") is not None,
                })
        except (json.JSONDecodeError, Exception):
            if stdout.strip():
                issues.append({"code": "PARSE_ERROR", "message": stdout[:500], "line": 0})

        return {
            "passed": rc == 0,
            "issue_count": len(issues),
            "issues": issues,
            "raw": stdout[:2000],
            "tool": "ruff",
        }
    finally:
        os.unlink(tmp)


def run_pylint(code: str, filepath: str = "code.py") -> dict:
    """Run pylint on a code string. Returns structured result."""
    tmp = write_temp_file(code)
    try:
        rc, stdout, stderr = _run(
            ["python", "-m", "pylint", tmp,
             "--output-format=json",
             "--disable=C0114,C0115,C0116,C0301,R0903",
             "--score=no"]
        )
        issues = []
        try:
            raw_list = json.loads(stdout) if stdout.strip().startswith("[") else []
            for item in raw_list:
                if item.get("type") in ("error", "warning", "convention"):
                    issues.append({
                        "code": item.get("message-id", ""),
                        "message": item.get("message", ""),
                        "line": item.get("line", 0),
                        "type": item.get("type", ""),
                    })
        except (json.JSONDecodeError, Exception):
            pass

        errors = [i for i in issues if i.get("type") == "error"]
        return {
            "passed": len(errors) == 0,
            "error_count": len(errors),
            "warning_count": len([i for i in issues if i.get("type") == "warning"]),
            "issues": issues,
            "raw": stdout[:2000],
            "tool": "pylint",
        }
    finally:
        os.unlink(tmp)


def run_bandit(code: str, filepath: str = "code.py") -> dict:
    """Run bandit security scanner. Returns structured result."""
    tmp = write_temp_file(code)
    try:
        rc, stdout, stderr = _run(
            ["python", "-m", "bandit", "-f", "json", "-q", tmp]
        )
        findings = []
        try:
            data = json.loads(stdout) if stdout.strip() else {}
            for result in data.get("results", []):
                findings.append({
                    "issue": result.get("issue_text", ""),
                    "severity": result.get("issue_severity", "LOW"),
                    "confidence": result.get("issue_confidence", "LOW"),
                    "line": result.get("line_number", 0),
                    "code": result.get("test_id", ""),
                    "cwe": result.get("issue_cwe", {}).get("id", ""),
                })
        except (json.JSONDecodeError, Exception):
            pass

        high_sev = [f for f in findings if f.get("severity") == "HIGH"]
        return {
            "passed": len(high_sev) == 0,
            "finding_count": len(findings),
            "high_severity": len(high_sev),
            "medium_severity": len([f for f in findings if f.get("severity") == "MEDIUM"]),
            "findings": findings,
            "raw": stdout[:2000],
            "tool": "bandit",
        }
    finally:
        os.unlink(tmp)


def run_syntax_check(code: str) -> dict:
    """Check Python syntax via compile()."""
    try:
        compile(code, "<forge_generated>", "exec")
        return {"passed": True, "error": None}
    except SyntaxError as e:
        return {"passed": False, "error": str(e), "line": e.lineno}


def run_pytest(
    test_code: str,
    impl_code: str,
    impl_filename: str = "implementation.py",
    all_generated_files: dict = None,
    project_dir_path: str = None,
) -> dict:
    """
    Run pytest against the generated implementation.

    Mirrors the full generated_project directory into a temp dir, writes
    a conftest.py that adds the project root to sys.path (so `from app.x import y`
    works without installing), then runs pytest.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        if all_generated_files and len(all_generated_files) > 1:
            # ── Mirror every generated file ───────────────────────────────────
            for rel_path, content in all_generated_files.items():
                dest = tmp / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content or "")

            # ── Ensure every directory is a Python package ────────────────────
            for d in sorted(tmp.rglob("*")):
                if d.is_dir() and not (d / "__init__.py").exists():
                    (d / "__init__.py").write_text("")

            # ── Write a project-level conftest.py that fixes sys.path ─────────
            # This is the key: without it, `from app.database import Base` fails
            # because pytest doesn't know where 'app' is.
            root_conftest = tmp / "conftest.py"
            if not root_conftest.exists():
                root_conftest.write_text(
                    "import sys, os\n"
                    "# Add the project root to sys.path so all app.* imports resolve\n"
                    f"sys.path.insert(0, '{tmpdir}')\n"
                )
            else:
                # Prepend path fix to existing conftest
                existing = root_conftest.read_text()
                root_conftest.write_text(
                    f"import sys\nsys.path.insert(0, '{tmpdir}')\n\n" + existing
                )

            # ── Install requirements if present ───────────────────────────────
            req_file = tmp / "requirements.txt"
            if req_file.exists():
                _run(
                    [sys.executable, "-m", "pip", "install", "-q",
                     "-r", str(req_file), "--quiet"],
                    timeout=120,
                )

            # ── Write the test file into a tests/ directory ───────────────────
            tests_dir = tmp / "tests"
            tests_dir.mkdir(exist_ok=True)
            (tests_dir / "__init__.py").write_text("")
            test_dest = tests_dir / "test_run.py"
            test_dest.write_text(test_code)
            test_target = str(test_dest)

        else:
            # ── Single-file fallback ──────────────────────────────────────────
            (tmp / impl_filename).write_text(impl_code)
            test_dest = tmp / "test_implementation.py"
            test_dest.write_text(test_code)
            # conftest.py adds tmpdir to sys.path
            (tmp / "conftest.py").write_text(
                f"import sys\nsys.path.insert(0, '{tmpdir}')\n"
            )
            test_target = str(test_dest)

        report_file = f"/tmp/forge_pytest_{os.getpid()}.json"
        rc, stdout, stderr = _run(
            [sys.executable, "-m", "pytest", test_target,
             "-v", "--tb=short",
             "--json-report", f"--json-report-file={report_file}",
             "--no-header", "-q",
             "-p", "no:cacheprovider",   # avoid .pytest_cache permission issues
             "--ignore=generated_project"],
            cwd=tmpdir,
            timeout=90,
        )

        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        test_details  = []
        try:
            with open(report_file) as f:
                report = json.load(f)
            summary       = report.get("summary", {})
            tests_passed  = summary.get("passed",  0)
            tests_failed  = summary.get("failed",  0) + summary.get("error", 0)
            tests_skipped = summary.get("skipped", 0)
            for test in report.get("tests", []):
                entry = {
                    "name":     test.get("nodeid", ""),
                    "outcome":  test.get("outcome", ""),
                    "duration": test.get("duration", 0),
                    "message":  "",
                }
                if test.get("outcome") in ("failed", "error"):
                    entry["message"] = str(
                        test.get("call", {}).get("longrepr", "")
                        or test.get("setup", {}).get("longrepr", "")
                    )[:500]
                test_details.append(entry)
        except Exception:
            # Fall back to regex parsing of stdout
            import re
            m = re.search(r"(\d+) passed", stdout)
            if m:
                tests_passed = int(m.group(1))
            m = re.search(r"(\d+) failed", stdout)
            if m:
                tests_failed = int(m.group(1))
            m = re.search(r"(\d+) skipped", stdout)
            if m:
                tests_skipped = int(m.group(1))
        finally:
            try:
                os.unlink(report_file)
            except Exception:
                pass

        # A run is "passed" if: at least 1 test ran AND none failed
        # Skipped tests (from pytest.skip()) don't count as failures
        all_skipped = (tests_passed == 0 and tests_skipped > 0 and tests_failed == 0)
        passed = (tests_failed == 0 and tests_passed > 0) or all_skipped

        return {
            "passed":        passed,
            "tests_passed":  tests_passed,
            "tests_failed":  tests_failed,
            "tests_skipped": tests_skipped,
            "all_skipped":   all_skipped,
            "test_details":  test_details,
            "stdout":        stdout[:3000],
            "stderr":        stderr[:1000],
            "return_code":   rc,
            "tool":          "pytest",
        }


def validate_file(code: str, filepath: str = "code.py") -> dict:
    """
    Run validation suite on a single file.

    Blocking failures: syntax errors, ruff E-codes (actual errors).
    Non-blocking: pylint warnings/conventions, ruff W-codes (warnings), unused imports.

    Design philosophy: we prefer false negatives (accepting slightly imperfect code)
    over false positives (blocking valid code that just uses modern library patterns).
    Pydantic v2 and SQLAlchemy 2.x trigger many pylint false positives — those are
    explicitly allowlisted so they don't block generation unnecessarily.
    """
    syntax = run_syntax_check(code)
    if not syntax["passed"]:
        return {
            "overall_passed": False,
            "syntax": syntax,
            "ruff": None,
            "pylint": None,
            "blocking_error": f"SyntaxError at line {syntax.get('line')}: {syntax.get('error')}",
        }

    ruff   = run_ruff(code, filepath)
    pylint = run_pylint(code, filepath)

    # ── Ruff blocking codes ────────────────────────────────────────────────────
    # E1xx-E9xx are real errors. W codes are stylistic warnings (non-blocking).
    # F401 unused import — non-blocking (common in __init__.py re-exports).
    # F811 redefinition — blocking (usually means a real logic error).
    BLOCKING_RUFF_PREFIXES = ("E1", "E2", "E3", "E4", "E5", "E7", "E9", "F8", "F9")
    NON_BLOCKING_RUFF = {
        "E501",  # line too long
        "W291", "W293", "W292", "W391",  # whitespace
        "F401",  # unused import (common in __init__.py)
        "F811",  # redefinition — actually let this through; pylint catches it better
        "B008",  # do not perform function calls in default args (false pos on Depends())
        "B006",  # do not use mutable data structures for argument defaults
    }

    ruff_blocking_issues = [
        i for i in ruff.get("issues", [])
        if (
            any(i.get("code", "").startswith(p) for p in BLOCKING_RUFF_PREFIXES)
            and i.get("code") not in NON_BLOCKING_RUFF
        )
    ]

    # ── Pylint blocking codes ──────────────────────────────────────────────────
    # Only E-codes (actual errors) are blocking. C/W/R (conventions/warnings/refactoring) are not.
    # Extended allowlist: many codes are false positives for modern library patterns.
    PYLINT_NON_BLOCKING = {
        "E0401",  # import-error — missing dep in test env, not a code bug
        "E0611",  # no-name-in-module — false positive for dynamic FastAPI/Pydantic exports
        "E0213",  # no-self-argument — false positive on Pydantic @field_validator (@classmethod)
        "E1101",  # has no member — false positive on SQLAlchemy column attributes
        "E1120",  # no-value-for-argument — false pos on FastAPI Depends() factories
        "E0102",  # function-redefined — false positive in test fixtures with same name
        "W0611",  # unused-import — non-blocking
        "W0612",  # unused-variable — non-blocking
        "W0613",  # unused-argument — non-blocking (common in FastAPI dependencies)
        "C0114", "C0115", "C0116",  # missing docstrings — non-blocking
        "C0301",  # line-too-long — non-blocking
        "R0903",  # too-few-public-methods — non-blocking (common in Pydantic schemas)
        "R0913",  # too-many-arguments — non-blocking
        "R0914",  # too-many-locals — non-blocking
    }

    pylint_errors = [
        i for i in pylint.get("issues", [])
        if i.get("type") == "error"
        and i.get("code", "").startswith("E")
        and i.get("code") not in PYLINT_NON_BLOCKING
    ]

    blocking = len(ruff_blocking_issues) > 0 or len(pylint_errors) > 0
    blocking_error = None
    if blocking:
        parts = []
        if ruff_blocking_issues:
            parts.append("Ruff: " + "; ".join(
                f"L{i['line']} {i['code']} {i['message']}"
                for i in ruff_blocking_issues[:3]
            ))
        if pylint_errors:
            parts.append("Pylint: " + "; ".join(
                f"L{i['line']} {i['message']}"
                for i in pylint_errors[:3]
            ))
        blocking_error = " | ".join(parts)

    return {
        "overall_passed":  not blocking,
        "syntax":          syntax,
        "ruff":            ruff,
        "pylint":          pylint,
        "blocking_error":  blocking_error,
        "ruff_blocking":   ruff_blocking_issues,
        "pylint_errors":   pylint_errors,
    }