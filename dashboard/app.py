"""
dashboard/app.py
Gradio dashboard for Forge — real-time reasoning traces, confidence scores,
test status, security findings, and W&B metrics.
"""
from __future__ import annotations
import json, threading, time
from pathlib import Path
from typing import Optional
import gradio as gr

# ── State shared between pipeline thread and dashboard ────────────────────────
_pipeline_state: dict = {}
_pipeline_running: bool = False
_pipeline_done: bool = False
_pipeline_thread: Optional[threading.Thread] = None


def get_state() -> dict:
    return _pipeline_state


def set_state(state: dict):
    global _pipeline_state
    _pipeline_state = dict(state)


# ── Dashboard refresh helpers ──────────────────────────────────────────────────

def _confidence_bar(score: int) -> str:
    if score >= 80:
        color = "#27AE60"
        label = "HIGH"
    elif score >= 60:
        color = "#F2994A"
        label = "MED"
    else:
        color = "#EB5757"
        label = "LOW"
    bar = "█" * (score // 10) + "░" * (10 - score // 10)
    return f'<span style="color:{color};font-weight:bold">[{bar}] {score}% {label}</span>'


def _render_decisions(audit: list) -> str:
    if not audit:
        return "<i>No decisions recorded yet.</i>"
    rows = []
    for d in audit[-8:]:  # Show last 8
        conf = d.get("confidence", 0)
        color = "#27AE60" if conf >= 80 else ("#F2994A" if conf >= 60 else "#EB5757")
        alts = "<br>".join(f"• {a}" for a in d.get("alternatives", [])[:3])
        rows.append(f"""
        <tr>
          <td style="padding:6px;font-weight:bold;color:#1B4F8A">{d.get('agent','')}</td>
          <td style="padding:6px;color:{color};font-weight:bold">{conf}%</td>
          <td style="padding:6px">{d.get('reasoning','')[:120]}</td>
          <td style="padding:6px;font-size:0.85em;color:#718096">{alts or '—'}</td>
        </tr>""")
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <tr style="background:#0D1B2A;color:white">
        <th style="padding:6px;text-align:left">Agent</th>
        <th style="padding:6px;text-align:left">Confidence</th>
        <th style="padding:6px;text-align:left">Reasoning</th>
        <th style="padding:6px;text-align:left">Alternatives</th>
      </tr>
      {"".join(rows)}
    </table>"""


def _render_tasks(task_plan: list) -> str:
    if not task_plan:
        return "<i>Task plan not yet generated.</i>"
    rows = []
    status_colors = {
        "done": "#27AE60", "failed": "#EB5757",
        "in_progress": "#2D9CDB", "pending": "#718096", "skipped": "#F2994A",
    }
    status_icons = {
        "done": "✅", "failed": "❌", "in_progress": "⚙️", "pending": "⏳", "skipped": "⏭️",
    }
    for t in task_plan:
        status = t.get("status", "pending")
        color = status_colors.get(status, "#718096")
        icon = status_icons.get(status, "•")
        rows.append(f"""
        <tr>
          <td style="padding:5px;color:#1B4F8A;font-weight:bold">{t.get('id','')}</td>
          <td style="padding:5px">{t.get('title','')}</td>
          <td style="padding:5px;color:{color};font-weight:bold">{icon} {status.upper()}</td>
          <td style="padding:5px;color:#718096">{t.get('risk_level','').upper()}</td>
        </tr>""")
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <tr style="background:#0D1B2A;color:white">
        <th style="padding:6px;text-align:left">ID</th>
        <th style="padding:6px;text-align:left">Task</th>
        <th style="padding:6px;text-align:left">Status</th>
        <th style="padding:6px;text-align:left">Risk</th>
      </tr>
      {"".join(rows)}
    </table>"""


def _render_tests(test_results: dict) -> str:
    if not test_results:
        return "<i>No tests run yet.</i>"
    rows = []
    for filepath, result in test_results.items():
        passed = result.get("tests_passed", 0)
        failed = result.get("tests_failed", 0)
        status = "✅ PASS" if result.get("passed") else "❌ FAIL"
        color = "#27AE60" if result.get("passed") else "#EB5757"
        rows.append(f"""
        <tr>
          <td style="padding:5px;font-family:monospace;font-size:0.85em">{filepath}</td>
          <td style="padding:5px;color:{color};font-weight:bold">{status}</td>
          <td style="padding:5px;color:#27AE60">{passed} passed</td>
          <td style="padding:5px;color:#EB5757">{failed} failed</td>
        </tr>""")
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <tr style="background:#0D1B2A;color:white">
        <th style="padding:6px;text-align:left">Test File</th>
        <th style="padding:6px;text-align:left">Result</th>
        <th style="padding:6px;text-align:left">Passed</th>
        <th style="padding:6px;text-align:left">Failed</th>
      </tr>
      {"".join(rows)}
    </table>"""


def _render_security(findings: list) -> str:
    if not findings:
        return '<span style="color:#27AE60;font-weight:bold">✅ No security findings.</span>'
    rows = []
    sev_colors = {"HIGH": "#EB5757", "MEDIUM": "#F2994A", "LOW": "#2D9CDB"}
    for f in findings:
        color = sev_colors.get(f.get("severity", "LOW"), "#718096")
        fix_status = "✅ Fixed" if f.get("fix_applied") else "⚠️ Open"
        fix_color = "#27AE60" if f.get("fix_applied") else "#F2994A"
        rows.append(f"""
        <tr>
          <td style="padding:5px;font-family:monospace;font-size:0.85em">{f.get('file','')}</td>
          <td style="padding:5px;color:{color};font-weight:bold">{f.get('severity','')}</td>
          <td style="padding:5px">{f.get('issue','')[:80]}</td>
          <td style="padding:5px;color:{fix_color};font-weight:bold">{fix_status}</td>
        </tr>""")
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9em">
      <tr style="background:#0D1B2A;color:white">
        <th style="padding:6px;text-align:left">File</th>
        <th style="padding:6px;text-align:left">Severity</th>
        <th style="padding:6px;text-align:left">Issue</th>
        <th style="padding:6px;text-align:left">Fix</th>
      </tr>
      {"".join(rows)}
    </table>"""


def _render_metrics(state: dict) -> str:
    metrics = state.get("agent_metrics", {})
    total_tokens = state.get("total_tokens", 0)
    api_calls = state.get("api_call_count", 0)
    phase = state.get("phase", "init")
    status = state.get("pipeline_status", "running")

    status_color = "#27AE60" if status == "completed" else ("#EB5757" if status == "failed" else "#2D9CDB")

    agent_rows = ""
    for agent, m in metrics.items():
        if not m:
            continue
        agent_rows += f"""
        <tr>
          <td style="padding:4px;font-weight:bold;color:#1B4F8A">{agent}</td>
          <td style="padding:4px">{m.get('latency_s', '—')}s</td>
          <td style="padding:4px">{m.get('tokens', '—')}</td>
          <td style="padding:4px">{m.get('confidence', '—')}{'%' if m.get('confidence') else ''}</td>
        </tr>"""

    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-bottom:16px">
      <div style="background:#EBF8FF;padding:12px;border-radius:8px;text-align:center">
        <div style="font-size:1.8em;font-weight:bold;color:#1B4F8A">{api_calls}</div>
        <div style="color:#718096;font-size:0.85em">API Calls</div>
      </div>
      <div style="background:#EBF8FF;padding:12px;border-radius:8px;text-align:center">
        <div style="font-size:1.8em;font-weight:bold;color:#1B4F8A">{total_tokens:,}</div>
        <div style="color:#718096;font-size:0.85em">Total Tokens</div>
      </div>
      <div style="background:#EBF8FF;padding:12px;border-radius:8px;text-align:center">
        <div style="font-size:1.4em;font-weight:bold;color:#1B4F8A">{phase.replace('_',' ').title()}</div>
        <div style="color:#718096;font-size:0.85em">Current Phase</div>
      </div>
      <div style="background:#EBF8FF;padding:12px;border-radius:8px;text-align:center">
        <div style="font-size:1.4em;font-weight:bold;color:{status_color}">{status.upper()}</div>
        <div style="color:#718096;font-size:0.85em">Status</div>
      </div>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:0.88em">
      <tr style="background:#0D1B2A;color:white">
        <th style="padding:5px;text-align:left">Agent</th>
        <th style="padding:5px;text-align:left">Latency</th>
        <th style="padding:5px;text-align:left">Tokens</th>
        <th style="padding:5px;text-align:left">Confidence</th>
      </tr>
      {agent_rows or '<tr><td colspan="4" style="padding:8px;text-align:center;color:#718096">Pipeline not started</td></tr>'}
    </table>"""


def _render_preview(state: dict) -> str:
    """
    Rich project preview:
    - Parses endpoints from actual generated code (not just architecture plan)
    - Interactive API demo cards — shows realistic request/response per endpoint
    - Project-specific run instructions derived from actual files
    - Data model schemas
    """
    files     = state.get("generated_files", {})
    arch      = state.get("architecture", {})
    stack     = state.get("inferred_stack", {})
    summary   = state.get("workflow_summary", {})

    if not files:
        return "<i>No files generated yet.</i>"

    py_files    = sorted(f for f in files if f.endswith(".py")
                         and "PLACEHOLDER" not in (files[f] or "")[:80])
    cfg_files   = sorted(f for f in files if not f.endswith(".py"))
    total_lines = sum(len((files[f] or "").splitlines()) for f in py_files)
    framework   = stack.get("framework", "FastAPI")
    is_fastapi  = "fastapi" in framework.lower()

    # Derive actual entrypoint from the file that contains `app = FastAPI()`
    entrypoint = "app.main"
    for f in sorted(py_files):
        code = files.get(f, "") or ""
        if "FastAPI()" in code or "= FastAPI(" in code:
            entrypoint = f.replace("/", ".").removesuffix(".py")
            break
    if entrypoint == "app.main":
        entrypoint = next(
            (f.replace("/", ".").removesuffix(".py") for f in py_files if f.endswith("main.py")),
            "app.main"
        )

    # ── Parse endpoints from actual generated code ────────────────────────────
    # Architecture plan may be empty/wrong — parse source as ground truth
    endpoints = _parse_endpoints_from_code(files, arch)
    models    = arch.get("data_models", [])

    # ── Stats bar ─────────────────────────────────────────────────────────────
    tasks_done  = summary.get("tasks_completed", len(py_files))
    tasks_total = summary.get("tasks_total", len(py_files))
    test_rate   = summary.get("test_pass_rate", 0)

    stats_bar = f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:20px">
      <div style="background:linear-gradient(135deg,#0D1B2A,#1B4F8A);padding:14px;border-radius:10px;text-align:center;color:white">
        <div style="font-size:1.8em;font-weight:700">{len(py_files)}</div>
        <div style="font-size:0.75em;opacity:0.8;margin-top:2px">Python Files</div>
      </div>
      <div style="background:linear-gradient(135deg,#134E4A,#047857);padding:14px;border-radius:10px;text-align:center;color:white">
        <div style="font-size:1.8em;font-weight:700">{total_lines:,}</div>
        <div style="font-size:0.75em;opacity:0.8;margin-top:2px">Lines of Code</div>
      </div>
      <div style="background:linear-gradient(135deg,#1E3A5F,#2D9CDB);padding:14px;border-radius:10px;text-align:center;color:white">
        <div style="font-size:1.8em;font-weight:700">{len(endpoints)}</div>
        <div style="font-size:0.75em;opacity:0.8;margin-top:2px">API Endpoints</div>
      </div>
      <div style="background:linear-gradient(135deg,#3D1A78,#7C3AED);padding:14px;border-radius:10px;text-align:center;color:white">
        <div style="font-size:1.8em;font-weight:700">{tasks_done}/{tasks_total}</div>
        <div style="font-size:0.75em;opacity:0.8;margin-top:2px">Tasks Done</div>
      </div>
      <div style="background:linear-gradient(135deg,#7A1D1D,#DC2626);padding:14px;border-radius:10px;text-align:center;color:white">
        <div style="font-size:1.8em;font-weight:700">{test_rate}%</div>
        <div style="font-size:0.75em;opacity:0.8;margin-top:2px">Tests Passing</div>
      </div>
    </div>"""

    # ── API Endpoint Explorer ─────────────────────────────────────────────────
    METHOD_STYLE = {
        "GET":    ("background:#0369A1;color:white",   "#EFF6FF", "#BFDBFE"),
        "POST":   ("background:#065F46;color:white",   "#F0FFF4", "#A7F3D0"),
        "PUT":    ("background:#92400E;color:white",   "#FFFBEB", "#FDE68A"),
        "PATCH":  ("background:#6B21A8;color:white",   "#FAF5FF", "#DDD6FE"),
        "DELETE": ("background:#991B1B;color:white",   "#FFF5F5", "#FECACA"),
    }

    endpoint_cards = ""
    for ep in endpoints:
        method   = ep.get("method", "GET").upper()
        path     = ep.get("path", "/")
        desc     = ep.get("description", _infer_description(method, path))
        fn_name  = ep.get("function", "")
        badge_style, card_bg, border_color = METHOD_STYLE.get(
            method, ("background:#374151;color:white", "#F9FAFB", "#E5E7EB"))

        # Build realistic sample request + response
        sample_req, sample_resp = _build_sample_exchange(method, path, ep, models, files)

        # curl snippet
        base = "http://localhost:8000"
        if method in ("POST", "PUT", "PATCH"):
            _sep = (",", ":")
            _compact = json.dumps(sample_req, separators=_sep) if sample_req else "{}"
            curl = f"curl -X {method} {base}{path} -H 'Content-Type: application/json' -d '{_compact}'"
        elif method == "DELETE":
            curl = f"curl -X DELETE {base}{path}"
        else:
            curl = f"curl {base}{path}"

        # Python snippet
        if method == "GET":
            py = (f"import requests\n"
                  f"r = requests.get('{base}{path}')\n"
                  f"print(r.json())")
        elif method == "DELETE":
            py = (f"import requests\n"
                  f"r = requests.delete('{base}{path}')\n"
                  f"print(r.status_code)  # 204 No Content")
        else:
            py = (f"import requests\n"
                  f"r = requests.{method.lower()}(\n"
                  f"    '{base}{path}',\n"
                  f"    json={json.dumps(sample_req, indent=4)}\n"
                  f")\n"
                  f"print(r.json())")

        resp_str = json.dumps(sample_resp, indent=2)
        status_code = {"GET": "200", "POST": "201", "PUT": "200",
                       "PATCH": "200", "DELETE": "204"}.get(method, "200")
        status_color = "#065F46" if method != "DELETE" else "#92400E"

        endpoint_cards += f"""
        <div style="border:1px solid {border_color};border-radius:10px;margin-bottom:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.06)">
          <div style="display:flex;align-items:center;gap:10px;padding:11px 16px;background:{card_bg}">
            <span style="{badge_style};font-weight:700;font-size:0.72em;padding:4px 10px;border-radius:5px;font-family:monospace;min-width:54px;text-align:center;letter-spacing:0.05em">{method}</span>
            <code style="font-weight:700;font-size:0.92em;color:#1A202C;letter-spacing:-0.01em">{path}</code>
            <span style="color:#64748B;font-size:0.82em;margin-left:6px">{desc}</span>
            {f'<span style="color:#94A3B8;font-size:0.75em;font-family:monospace;margin-left:auto">fn: {fn_name}</span>' if fn_name else ''}
          </div>
          <div style="background:#FAFAFA;border-top:1px solid {border_color}">
            <details style="margin:0">
              <summary style="padding:8px 16px;cursor:pointer;color:#1B4F8A;font-size:0.8em;font-weight:600;user-select:none;list-style:none;display:flex;align-items:center;gap:6px">
                <span>▶ See request &amp; response example</span>
                <span style="background:#E0F2FE;color:#0369A1;padding:1px 7px;border-radius:8px;font-size:0.85em">{status_code}</span>
              </summary>
              <div style="padding:12px 16px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;border-top:1px solid {border_color}">
                <div>
                  <div style="font-size:0.72em;color:#64748B;font-weight:700;margin-bottom:5px;text-transform:uppercase;letter-spacing:0.05em">Request Body</div>
                  <pre style="background:#0F172A;color:#E2E8F0;padding:10px;border-radius:7px;font-size:0.78em;margin:0;overflow-x:auto;white-space:pre-wrap;min-height:60px">{json.dumps(sample_req, indent=2) if sample_req else "—  (no body)"}</pre>
                </div>
                <div>
                  <div style="font-size:0.72em;color:#64748B;font-weight:700;margin-bottom:5px;text-transform:uppercase;letter-spacing:0.05em">Response <span style="color:{status_color};font-weight:800">{status_code}</span></div>
                  <pre style="background:#0F172A;color:#86EFAC;padding:10px;border-radius:7px;font-size:0.78em;margin:0;overflow-x:auto;white-space:pre-wrap;min-height:60px">{resp_str}</pre>
                </div>
                <div>
                  <div style="font-size:0.72em;color:#64748B;font-weight:700;margin-bottom:5px;text-transform:uppercase;letter-spacing:0.05em">curl</div>
                  <pre style="background:#0F172A;color:#FCD34D;padding:10px;border-radius:7px;font-size:0.75em;margin:0;overflow-x:auto;white-space:pre-wrap;min-height:60px">{curl}</pre>
                  <div style="font-size:0.72em;color:#64748B;font-weight:700;margin:6px 0 4px;text-transform:uppercase;letter-spacing:0.05em">Python</div>
                  <pre style="background:#0F172A;color:#93C5FD;padding:10px;border-radius:7px;font-size:0.75em;margin:0;overflow-x:auto;white-space:pre-wrap">{py}</pre>
                </div>
              </div>
            </details>
          </div>
        </div>"""

    swagger_note = (
        f'<span style="font-size:0.8em;color:#64748B;font-weight:400;margin-left:8px">'
        f'· Interactive Swagger UI at '
        f'<a href="http://localhost:8000/docs" target="_blank" '
        f'style="color:#2D9CDB;text-decoration:none;font-family:monospace">'
        f'http://localhost:8000/docs</a> when running</span>'
    ) if is_fastapi else ""

    endpoints_section = f"""
    <div style="margin-bottom:22px">
      <div style="font-weight:700;color:#0D1B2A;font-size:1em;margin-bottom:10px;display:flex;align-items:center;gap:6px">
        🔌 API Endpoints
        <span style="background:#EBF8FF;color:#1B4F8A;font-size:0.75em;padding:2px 9px;border-radius:10px;font-weight:700">{len(endpoints)} routes</span>
        {swagger_note}
      </div>
      {endpoint_cards if endpoint_cards else
       '<div style="padding:16px;background:#FFF7ED;border:1px solid #FED7AA;border-radius:8px;color:#92400E;font-size:0.85em">⚠ Could not extract endpoints from generated code. Check the router files manually.</div>'}
    </div>"""

    # ── Data Models ───────────────────────────────────────────────────────────
    # Also parse from generated code if architecture is empty
    if not models:
        models = _parse_models_from_code(files)

    model_cards = ""
    for m in models[:5]:
        name   = m.get("name", "Model")
        fields = m.get("fields", [])
        rows   = "".join(
            f'<tr style="border-top:1px solid #F1F5F9">'
            f'<td style="padding:5px 10px;font-family:monospace;font-size:0.8em;color:#1B4F8A;font-weight:600">{f.get("name","")}</td>'
            f'<td style="padding:5px 10px;font-size:0.8em;color:#7C3AED">{f.get("type","str")}</td>'
            f'<td style="padding:5px 10px;font-size:0.78em;color:#64748B">'
            f'{"required" if not f.get("optional") else "optional"}'
            f'{"· PK" if f.get("name","") == "id" else ""}'
            f'</td></tr>'
            for f in fields[:10]
        )
        model_cards += f"""
        <div style="border:1px solid #E2E8F0;border-radius:9px;overflow:hidden;min-width:180px;flex:1">
          <div style="background:#0D1B2A;color:white;padding:9px 13px;font-weight:700;font-size:0.85em;font-family:monospace;letter-spacing:0.03em">{name}</div>
          <table style="width:100%;border-collapse:collapse">
            <tr style="background:#F8FAFC"><th style="padding:4px 10px;text-align:left;font-size:0.72em;color:#94A3B8;font-weight:700">FIELD</th><th style="padding:4px 10px;text-align:left;font-size:0.72em;color:#94A3B8;font-weight:700">TYPE</th><th style="padding:4px 10px;text-align:left;font-size:0.72em;color:#94A3B8;font-weight:700">INFO</th></tr>
            {rows or '<tr><td colspan="3" style="padding:8px 10px;color:#94A3B8;font-size:0.8em">No fields parsed</td></tr>'}
          </table>
        </div>"""

    models_section = f"""
    <div style="margin-bottom:22px">
      <div style="font-weight:700;color:#0D1B2A;font-size:1em;margin-bottom:10px">🗃️ Data Models</div>
      <div style="display:flex;flex-wrap:wrap;gap:10px">{model_cards}</div>
    </div>""" if model_cards else ""

    # ── File tree + Project-specific run instructions ─────────────────────────
    role_icons = {
        "main":       "🏠", "database": "🗄️", "model":    "📐",
        "schema":     "📋", "router":   "🔀", "auth":     "🔐",
        "security":   "🔐", "config":   "⚙️", "test":     "🧪",
        "dockerfile": "🐳", "docker":   "🐳", "require":  "📦",
        "crud":       "🔧", "deps":     "⛓️",
    }
    tree_rows = ""
    for f in py_files + cfg_files:
        lines = len((files[f] or "").splitlines())
        fname = Path(f).name.lower()
        icon  = next((v for k, v in role_icons.items() if k in fname), "📄")
        bg    = "#F7FAFC" if f.endswith(".py") else "#FFFBEB"
        tree_rows += (
            f'<tr style="background:{bg};border-bottom:1px solid #F1F5F9">'
            f'<td style="padding:5px 10px;font-family:monospace;font-size:0.8em;color:#1A202C">{icon} {f}</td>'
            f'<td style="padding:5px 10px;color:#94A3B8;font-size:0.78em;text-align:right">{lines} lines</td>'
            f'</tr>'
        )

    # Build project-specific run instructions
    req_file  = files.get("requirements.txt", "")
    env_file  = files.get(".env.example", files.get(".env", ""))
    has_alembic = any("alembic" in (files.get(f) or "") for f in py_files)
    has_db_init = any("init_db" in (files.get(f) or "") for f in py_files)
    db_type   = stack.get("database", "SQLite").lower()
    port      = 8000

    run_steps = []
    run_steps.append(('# Install dependencies', f'pip install -r requirements.txt'))
    if env_file:
        run_steps.append(('# Copy environment config', 'cp .env.example .env\n# Edit .env with your settings'))
    if has_alembic:
        run_steps.append(('# Run database migrations', 'alembic upgrade head'))
    elif has_db_init and "sqlite" in db_type:
        run_steps.append(('# Database auto-creates on first run', '# (SQLite — no setup needed)'))
    elif "postgres" in db_type:
        run_steps.append(('# Start PostgreSQL first', 'docker compose up db -d'))
    if is_fastapi:
        run_steps.append(('# Start the API server', f'uvicorn {entrypoint}:app --reload --port {port}'))
    else:
        run_steps.append(('# Run the app', f'python -m {entrypoint.replace(".", "/")}'))
    if is_fastapi:
        run_steps.append(('# Open interactive API docs', f'http://localhost:{port}/docs'))

    run_html_lines = ""
    for comment, cmd in run_steps:
        run_html_lines += f'<span style="color:#4ADE80;font-size:0.82em">{comment}</span>\n{cmd}\n\n'

    pkgs = [l.strip().split(">=")[0].split("==")[0].split("[")[0]
            for l in req_file.splitlines()
            if l.strip() and not l.startswith("#")] if req_file else []
    pkg_chips = " ".join(
        f'<span style="background:#EDF2F7;color:#1B4F8A;padding:2px 8px;border-radius:10px;font-size:0.76em;display:inline-block;margin:2px;border:1px solid #BEE3F8">{p}</span>'
        for p in pkgs[:14]
    )

    file_run_section = f"""
    <div style="display:grid;grid-template-columns:1.2fr 1fr;gap:16px;margin-bottom:20px">
      <div>
        <div style="font-weight:700;color:#0D1B2A;margin-bottom:8px">📁 File Tree
          <span style="font-size:0.75em;color:#94A3B8;font-weight:400;margin-left:6px">{len(py_files)} Python · {len(cfg_files)} config · {total_lines:,} lines</span>
        </div>
        <table style="width:100%;border-collapse:collapse;border:1px solid #E2E8F0;border-radius:8px;overflow:hidden">
          {tree_rows}
        </table>
        <div style="margin-top:10px">{pkg_chips}</div>
      </div>
      <div>
        <div style="font-weight:700;color:#0D1B2A;margin-bottom:8px">🚀 Run this project</div>
        <div style="background:#0F172A;color:#E2E8F0;padding:14px 16px;border-radius:9px;font-family:monospace;font-size:0.8em;line-height:2;white-space:pre-wrap">{run_html_lines.rstrip()}</div>
        <div style="margin-top:10px">
          <div style="font-weight:700;color:#0D1B2A;font-size:0.85em;margin-bottom:5px">🐳 Or with Docker</div>
          <div style="background:#0F172A;color:#E2E8F0;padding:10px 14px;border-radius:9px;font-family:monospace;font-size:0.8em;line-height:2">
            <span style="color:#4ADE80;font-size:0.82em"># Builds &amp; starts all services</span>
            docker compose up --build
            <span style="color:#4ADE80;font-size:0.82em"># API at http://localhost:{port}</span>
          </div>
        </div>
        <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:5px">
          {"".join(f'<span style="background:#EBF8FF;color:#1B4F8A;padding:3px 10px;border-radius:12px;font-size:0.76em;border:1px solid #BEE3F8">{k}: {v}</span>' for k, v in stack.items())}
        </div>
      </div>
    </div>"""

    return stats_bar + endpoints_section + models_section + file_run_section


def _parse_endpoints_from_code(files: dict, arch: dict) -> list[dict]:
    """
    Parse HTTP endpoints from actual generated router/main files.
    Falls back to architecture plan if parsing yields nothing.
    Looks for @router.get/post/put/delete and @app.get/post/put/delete decorators.
    Also reads the router prefix from include_router() calls to reconstruct full paths.
    """
    import re

    # First collect router prefix mappings from main.py
    prefix_map: dict[str, str] = {}
    for fpath, code in files.items():
        if not code or not fpath.endswith(".py"):
            continue
        # Match: app.include_router(items.router, prefix="/items")
        for m in re.finditer(
            r'include_router\s*\(\s*(\w+)(?:\.\w+)?\s*(?:,.*?prefix\s*=\s*["\']([^"\']+)["\'])?',
            code,
        ):
            mod_name = m.group(1)
            prefix   = m.group(2) or ""
            prefix_map[mod_name] = prefix

        # Match: app.include_router(router, prefix="/expenses", tags=["expenses"])
        for m in re.finditer(
            r'include_router\s*\(\s*router\s*,\s*prefix\s*=\s*["\']([^"\']+)["\']',
            code,
        ):
            # associate with the file's own stem
            stem = Path(fpath).stem
            prefix_map[stem] = m.group(1)
            prefix_map["router"] = m.group(1)

    endpoints: list[dict] = []
    DECORATOR_RE = re.compile(
        r'@(?:router|app)\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']*)["\']'
        r'(?:.*?(?:summary|description)\s*=\s*["\']([^"\']*)["\'])?',
        re.IGNORECASE,
    )
    FN_RE = re.compile(r'async def (\w+)|def (\w+)')

    for fpath, code in files.items():
        if not code or not fpath.endswith(".py"):
            continue
        # Skip test files
        if "test" in Path(fpath).name.lower():
            continue

        stem   = Path(fpath).stem
        prefix = prefix_map.get(stem, prefix_map.get("router", ""))

        lines = code.splitlines()
        for i, line in enumerate(lines):
            dm = DECORATOR_RE.search(line)
            if not dm:
                continue
            method = dm.group(1).upper()
            path   = dm.group(2)
            desc   = dm.group(3) or ""

            # Full path = router prefix + route path
            full_path = (prefix.rstrip("/") + "/" + path.lstrip("/")).rstrip("/")
            if not full_path.startswith("/"):
                full_path = "/" + full_path

            # Find the function name on the next non-decorator line
            fn_name = ""
            for j in range(i + 1, min(i + 5, len(lines))):
                fm = FN_RE.search(lines[j])
                if fm:
                    fn_name = fm.group(1) or fm.group(2)
                    break

            # Infer description from function name if not found in decorator
            if not desc:
                desc = _infer_description(method, full_path, fn_name)

            endpoints.append({
                "method":   method,
                "path":     full_path,
                "description": desc,
                "function": fn_name,
                "source":   fpath,
            })

    # Deduplicate by method+path
    seen: set[str] = set()
    unique: list[dict] = []
    for ep in endpoints:
        key = f"{ep['method']}:{ep['path']}"
        if key not in seen:
            seen.add(key)
            unique.append(ep)

    # Fall back to architecture plan if we found nothing
    if not unique:
        unique = arch.get("api_endpoints", [])

    return sorted(unique, key=lambda e: (e.get("path", ""), e.get("method", "")))


def _infer_description(method: str, path: str, fn_name: str = "") -> str:
    """Generate a human-readable description from method + path when none is available."""
    segments = [s for s in path.split("/") if s and not s.startswith("{")]
    resource = segments[-1].replace("_", " ") if segments else "resource"
    is_detail = "{" in path
    MAP = {
        "GET":    f"Get {'a specific ' + resource if is_detail else 'all ' + resource}",
        "POST":   f"Create a new {resource.rstrip('s')}",
        "PUT":    f"Update a {resource.rstrip('s')}",
        "PATCH":  f"Partially update a {resource.rstrip('s')}",
        "DELETE": f"Delete a {resource.rstrip('s')}",
    }
    return MAP.get(method, fn_name.replace("_", " ").capitalize())


def _parse_models_from_code(files: dict) -> list[dict]:
    """
    Parse SQLAlchemy models and Pydantic schemas from generated source files.
    Returns a list of {name, fields:[{name, type}]} dicts.
    """
    import re
    models: list[dict] = []
    seen_names: set[str] = set()

    CLASS_RE  = re.compile(r'^class (\w+)\s*\(([^)]+)\)\s*:', re.MULTILINE)
    FIELD_RE  = re.compile(
        r'^\s{4}(\w+)\s*(?::\s*(?:Mapped\[)?([^\]=\n]+?)(?:\])?)?\s*(?:=\s*mapped_column|=\s*Column|\s*$)',
        re.MULTILINE,
    )
    PYDANTIC_FIELD_RE = re.compile(
        r'^\s{4}(\w+)\s*:\s*(Optional\[)?([^\n=]+?)(?:\])?\s*(?:=|$)',
        re.MULTILINE,
    )

    for fpath, code in files.items():
        if not code or not fpath.endswith(".py") or "test" in fpath.lower():
            continue

        for cm in CLASS_RE.finditer(code):
            cls_name  = cm.group(1)
            bases     = cm.group(2)
            if cls_name in seen_names or cls_name.startswith("_"):
                continue

            is_model  = "Base" in bases or "DeclarativeBase" in bases
            is_schema = "BaseModel" in bases

            if not (is_model or is_schema):
                continue

            # Extract the class body
            start = cm.end()
            # Find next class definition or end of file
            next_class = CLASS_RE.search(code, start)
            body = code[start:next_class.start() if next_class else len(code)]

            fields: list[dict] = []
            if is_model:
                for fm in FIELD_RE.finditer(body):
                    fname = fm.group(1)
                    ftype = (fm.group(2) or "Any").strip().strip("\"'")
                    if fname.startswith("_") or fname in ("metadata", "registry"):
                        continue
                    fields.append({"name": fname, "type": ftype, "optional": False})
            else:
                for fm in PYDANTIC_FIELD_RE.finditer(body):
                    fname    = fm.group(1)
                    optional = bool(fm.group(2))
                    ftype    = fm.group(3).strip().strip("\"'")
                    if fname.startswith("_") or fname == "model_config":
                        continue
                    fields.append({"name": fname, "type": ftype, "optional": optional})

            if fields or is_model:
                seen_names.add(cls_name)
                models.append({"name": cls_name, "fields": fields})

    return models[:6]


def _build_sample_exchange(
    method: str, path: str, ep: dict, models: list, files: dict
) -> tuple[dict | None, dict | list | str]:
    """
    Build a realistic request body + response example for an endpoint.
    Tries to match the endpoint path to a known model to use its real fields.
    """
    # Try to find a matching model by path segment
    segments = [s.lower() for s in path.split("/") if s and not s.startswith("{")]
    resource = segments[-1] if segments else ""
    resource_singular = resource.rstrip("s")

    matched_model: dict | None = None
    for m in models:
        mname = m.get("name", "").lower()
        if mname == resource_singular or mname == resource or mname in resource or resource in mname:
            matched_model = m
            break

    # Also try parsing from source if no models in architecture
    if not matched_model and files:
        parsed = _parse_models_from_code(files)
        for m in parsed:
            mname = m.get("name", "").lower()
            if mname == resource_singular or mname == resource or resource in mname:
                matched_model = m
                break
        if not matched_model and parsed:
            matched_model = parsed[0]

    def field_example(fname: str, ftype: str) -> object:
        ftype = ftype.lower()
        fname = fname.lower()
        if fname == "id":                return 1
        if "email" in fname:             return "user@example.com"
        if "password" in fname or "hash" in fname: return "secret123"
        if "amount" in fname or "price" in fname or "cost" in fname: return 29.99
        if "date" in fname:              return "2025-04-18"
        if "bool" in ftype or fname.startswith("is_") or fname.startswith("has_"): return True
        if "int" in ftype:               return 1
        if "float" in ftype:             return 1.5
        if "list" in ftype:              return []
        return f"example_{fname}"

    # Build request body for write methods
    req_body: dict | None = None
    if method in ("POST", "PUT", "PATCH") and matched_model:
        req_body = {
            f["name"]: field_example(f["name"], f.get("type", "str"))
            for f in matched_model.get("fields", [])
            if f["name"] not in ("id", "created_at", "updated_at")
            and not f.get("name", "").startswith("_")
        }
        if not req_body:
            req_body = {"name": f"example_{resource_singular}", "value": 1}
    elif method in ("POST", "PUT", "PATCH"):
        req_body = {"name": f"example_{resource_singular}", "value": 1}

    # Build response example
    if method == "DELETE":
        return req_body, "204 No Content"

    if matched_model:
        single = {f["name"]: field_example(f["name"], f.get("type", "str"))
                  for f in matched_model.get("fields", [])}
        single.setdefault("id", 1)
        if method == "GET" and "{" not in path:
            return req_body, [single]  # list endpoint returns array
        return req_body, single
    else:
        # Minimal generic response
        return req_body, {"id": 1, "status": "ok"}


def _make_zip() -> str:
    import shutil
    from pathlib import Path
    project_dir = Path("./generated_project")
    if not project_dir.exists():
        return None
    zip_path = "/tmp/forge_generated_project"
    shutil.make_archive(zip_path, "zip", ".", "generated_project")
    return zip_path + ".zip"



def _render_summary(summary: dict) -> str:
    if not summary:
        return "<i>Pipeline not complete yet.</i>"
    return f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
      <div style="background:#F0FFF4;padding:10px;border-radius:8px;text-align:center;border:1px solid #27AE60">
        <div style="font-size:2em;font-weight:bold;color:#27AE60">{summary.get('tasks_completed',0)}/{summary.get('tasks_total',0)}</div>
        <div style="color:#718096">Tasks Completed</div>
      </div>
      <div style="background:#EBF8FF;padding:10px;border-radius:8px;text-align:center;border:1px solid #2D9CDB">
        <div style="font-size:2em;font-weight:bold;color:#2D9CDB">{summary.get('files_generated',0)}</div>
        <div style="color:#718096">Files Generated</div>
      </div>
      <div style="background:#F0FFF4;padding:10px;border-radius:8px;text-align:center;border:1px solid #27AE60">
        <div style="font-size:2em;font-weight:bold;color:#27AE60">{summary.get('test_pass_rate',0)}%</div>
        <div style="color:#718096">Test Pass Rate</div>
      </div>
      <div style="background:#FFF5F5;padding:10px;border-radius:8px;text-align:center;border:1px solid #EB5757">
        <div style="font-size:2em;font-weight:bold;color:#EB5757">{summary.get('security_findings_total',0)}</div>
        <div style="color:#718096">Security Findings</div>
      </div>
      <div style="background:#F0FFF4;padding:10px;border-radius:8px;text-align:center;border:1px solid #27AE60">
        <div style="font-size:2em;font-weight:bold;color:#27AE60">{summary.get('security_findings_fixed',0)}</div>
        <div style="color:#718096">Findings Fixed</div>
      </div>
      <div style="background:#EBF8FF;padding:10px;border-radius:8px;text-align:center;border:1px solid #2D9CDB">
        <div style="font-size:2em;font-weight:bold;color:#1B4F8A">{summary.get('api_calls_total',0)}</div>
        <div style="color:#718096">API Calls Total</div>
      </div>
    </div>"""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def create_dashboard():
    """Build and return the Gradio Blocks app."""

    with gr.Blocks(
        title="Forge — Engineering Intelligence System",
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="orange",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .forge-header { background: #0D1B2A; padding: 20px; border-radius: 8px; margin-bottom: 16px; }
        .forge-header h1 { color: #2D9CDB; margin: 0; font-size: 2em; }
        .forge-header p { color: #A0AEC0; margin: 4px 0 0 0; }
        .section-label { color: #0D1B2A; font-weight: bold; font-size: 1.05em; margin-bottom: 4px; }
        """,
    ) as demo:

        # Header
        gr.HTML("""
        <div class="forge-header">
          <h1>⚙️ FORGE</h1>
          <p>Engineering Intelligence System — Agentic AI Framework</p>
        </div>""")

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Project Specification",
                placeholder="Describe the software project you want to build...\nExample: Build a REST API for a personal expense tracker with categories, tags, and monthly budget limits.",
                lines=4,
                scale=4,
            )
            with gr.Column(scale=1):
                run_btn = gr.Button(
                    "🚀 Run Forge", variant="primary", size="lg",
                    interactive=False,   # disabled until prompt is entered
                )
                stop_btn = gr.Button(
                    "⏹ Stop", variant="stop", size="sm",
                    interactive=False,   # disabled until pipeline is running
                )
                status_badge = gr.HTML('<span style="color:#718096">Ready — enter a prompt to begin</span>')

        # Priority weights
        with gr.Accordion("⚙️ Priority Weights (optional)", open=False):
            with gr.Row():
                w_speed    = gr.Slider(0, 1, value=0.2,  step=0.05, label="Generation Speed")
                w_quality  = gr.Slider(0, 1, value=0.25, step=0.05, label="Code Quality")
                w_tests    = gr.Slider(0, 1, value=0.25, step=0.05, label="Test Coverage")
                w_security = gr.Slider(0, 1, value=0.2,  step=0.05, label="Security Hardness")
                w_simple   = gr.Slider(0, 1, value=0.1,  step=0.05, label="Simplicity")

        gr.HTML("<hr>")

        # Tabs
        with gr.Tabs():

            with gr.TabItem("📊 Live Metrics"):
                metrics_html = gr.HTML("<i>Start the pipeline to see metrics.</i>")

            with gr.TabItem("🧠 Decision Audit Trail"):
                gr.HTML('<p style="color:#718096;font-size:0.9em">Every architectural decision — what was considered, why it was chosen, confidence score.</p>')
                decisions_html = gr.HTML("<i>No decisions yet.</i>")

            with gr.TabItem("📋 Task Plan"):
                tasks_html = gr.HTML("<i>Task plan not generated yet.</i>")

            with gr.TabItem("🧪 Test Results"):
                tests_html = gr.HTML("<i>No tests run yet.</i>")

            with gr.TabItem("🔐 Security"):
                security_html = gr.HTML("<i>Security audit not run yet.</i>")

            with gr.TabItem("🏗️ Architecture"):
                arch_html = gr.HTML("<i>Architecture not generated yet.</i>")

            with gr.TabItem("📁 Files & Editor"):
                with gr.Row():
                    file_selector = gr.Dropdown(
                        label="Select file", choices=[], interactive=True, scale=3
                    )
                    download_btn = gr.Button("⬇ Download ZIP", size="sm", scale=1)
                file_content = gr.Textbox(
                    label="File content (editable — select a file above)",
                    lines=25,
                    max_lines=60,
                    interactive=True,
                )
                with gr.Row():
                    save_btn   = gr.Button("💾 Save changes", variant="primary", size="sm",
                                           interactive=False)
                    save_status = gr.HTML("")
                gr.HTML('<hr style="margin:8px 0">')
                gr.HTML('<div style="font-weight:bold;color:#0D1B2A;margin-bottom:4px">✏️ AI Edit — describe a change to apply to the selected file</div>')
                with gr.Row():
                    edit_prompt = gr.Textbox(
                        placeholder='e.g. "Add a /expenses/summary endpoint that returns total by category"',
                        label="", scale=4, lines=2,
                    )
                    ai_edit_btn = gr.Button("Apply change", variant="secondary", size="sm", scale=1)
                ai_edit_status = gr.HTML("")
                download_file  = gr.File(label="Download", visible=False)

            with gr.TabItem("🚀 Project Preview"):
                preview_html = gr.HTML("<i>Run the pipeline to see the project preview.</i>")

            with gr.TabItem("✅ Summary"):
                summary_html = gr.HTML("<i>Pipeline not complete.</i>")

        # Error banner — shown on crash, hidden otherwise
        error_banner = gr.HTML("", visible=False)

        # Activity log
        with gr.Accordion("📜 Activity Log", open=False):
            activity_log = gr.Textbox(
                label="",
                lines=12,
                max_lines=20,
                interactive=False,
            )

        # ── Event handlers ────────────────────────────────────────────────────

        # Enable Run button only when prompt has content
        def on_prompt_change(text):
            has_text = bool(text and text.strip())
            return gr.update(interactive=has_text)

        prompt_input.change(fn=on_prompt_change, inputs=[prompt_input], outputs=[run_btn])

        def stop_pipeline():
            global _pipeline_running, _pipeline_done
            _pipeline_running = False
            _pipeline_done = True
            log_action("Dashboard", "Stop requested by user")
            return (
                '<span style="color:#F2994A;font-weight:bold">⏹ Stopped by user</span>',
                gr.update(interactive=True),   # run_btn
                gr.update(interactive=False),  # stop_btn
            )

        stop_btn.click(
            fn=stop_pipeline,
            inputs=[],
            outputs=[status_badge, run_btn, stop_btn],
        )

        def run_pipeline(prompt, w_spd, w_qual, w_tst, w_sec, w_smp):
            global _pipeline_running, _pipeline_done, _pipeline_thread, _pipeline_state

            if not prompt.strip():
                yield (
                    '<span style="color:#EB5757">⚠️ Please enter a project specification.</span>',
                    gr.update(interactive=True),   # run_btn
                    gr.update(interactive=False),  # stop_btn
                    gr.update(visible=False),      # error_banner
                    *([gr.update()] * 8),          # 8 tab html outputs
                    "",                            # activity_log
                    gr.update(choices=[]),         # file_selector
                )
                return

            _pipeline_running = True
            _pipeline_done = False
            _pipeline_state = {}

            priority_weights = {
                "speed": w_spd, "quality": w_qual,
                "test_coverage": w_tst, "security": w_sec, "simplicity": w_smp,
            }

            def _run():
                global _pipeline_state, _pipeline_done, _pipeline_running
                try:
                    from core.graph import build_graph, get_initial_state
                    from core.observability import init_wandb, init_langsmith, init_log
                    import os, yaml
                    from pathlib import Path

                    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))

                    init_log(cfg["output"]["log_file"])
                    if os.environ.get("WANDB_API_KEY"):
                        init_wandb(cfg["observability"]["wandb_project"], {
                            "prompt": prompt[:200],
                            "priority_weights": priority_weights,
                        })
                    if os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"):
                        init_langsmith(cfg["observability"]["langsmith_project"])

                    initial = get_initial_state(prompt, priority_weights)
                    graph = build_graph()

                    for step in graph.stream(initial):
                        if not _pipeline_running:  # stop button was pressed
                            break
                        for node_name, node_state in step.items():
                            set_state(node_state)

                    _pipeline_done = True

                except Exception as e:
                    import traceback as tb
                    err_msg = str(e)
                    trace = tb.format_exc()
                    friendly = _classify_crash(err_msg)
                    current = dict(_pipeline_state)
                    current["pipeline_status"] = "failed"
                    current["crash_error"] = err_msg
                    current["crash_traceback"] = trace
                    current["crash_friendly"] = friendly
                    set_state(current)
                    _pipeline_done = True

                finally:
                    _pipeline_running = False

            _pipeline_thread = threading.Thread(target=_run, daemon=True)
            _pipeline_thread.start()

            # Poll and yield updates every second
            for _ in range(600):  # max 10 minutes
                time.sleep(1)
                state = get_state()

                phase  = state.get("phase", "init")
                status = state.get("pipeline_status", "running")
                crash  = state.get("crash_error")
                done   = _pipeline_done

                # ── Button states ─────────────────────────────────────────────
                run_btn_update  = gr.update(interactive=done)
                stop_btn_update = gr.update(interactive=not done)

                # ── Status badge ──────────────────────────────────────────────
                if status == "completed":
                    status_html = '<span style="color:#27AE60;font-weight:bold">✅ Completed</span>'
                elif status == "failed":
                    status_html = '<span style="color:#EB5757;font-weight:bold">❌ Failed</span>'
                elif not _pipeline_running and done:
                    status_html = '<span style="color:#F2994A;font-weight:bold">⏹ Stopped</span>'
                else:
                    status_html = (
                        f'<span style="color:#2D9CDB;font-weight:bold">'
                        f'⚙️ {phase.replace("_", " ").title()}'
                        f'</span>'
                    )

                # ── Error banner ──────────────────────────────────────────────
                if crash:
                    friendly = state.get("crash_friendly", "An unexpected error occurred.")
                    trace_lines = (state.get("crash_traceback", "").strip().split("\n"))
                    trace_display = "\n".join(trace_lines[-6:])
                    err_html = f"""
                    <div style="background:#FFF5F5;border:2px solid #EB5757;border-radius:8px;
                                padding:16px;margin:8px 0;">
                      <div style="font-size:1.1em;font-weight:bold;color:#EB5757;margin-bottom:8px;">
                        ❌ Pipeline crashed
                      </div>
                      <div style="color:#C53030;font-weight:bold;margin-bottom:8px;">{friendly}</div>
                      <details>
                        <summary style="cursor:pointer;color:#718096;font-size:0.9em;">
                          Show technical details
                        </summary>
                        <pre style="background:#1A202C;color:#FC8181;padding:10px;border-radius:4px;
                                    font-size:0.8em;overflow-x:auto;margin-top:8px;">{trace_display}</pre>
                        <div style="color:#718096;font-size:0.85em;margin-top:4px;">
                          Full error: {str(crash)[:300]}
                        </div>
                      </details>
                      <div style="margin-top:12px;font-size:0.9em;color:#4A5568;">
                        💡 <b>What to try:</b> Check the Activity Log for the last successful step.
                        Any files generated so far are still in the Generated Files tab.
                      </div>
                    </div>"""
                    err_update = gr.update(value=err_html, visible=True)
                else:
                    err_update = gr.update(visible=False)

                # ── Activity log ──────────────────────────────────────────────
                from core.observability import get_log_buffer
                log_lines = "\n".join(
                    f"[{e.get('timestamp','')[-8:-1]}] [{e.get('level','INFO')}]"
                    f" [{e.get('agent','')}] {e.get('action','')}"
                    + (f" — {e.get('detail','')[:80]}" if e.get('detail') else "")
                    for e in get_log_buffer()[-40:]
                )

                # ── Architecture tab ──────────────────────────────────────────
                arch = state.get("architecture", {})
                arch_text = f"<b>Pattern:</b> {state.get('chosen_pattern', '—')}<br><br>"
                if arch.get("key_decisions"):
                    arch_text += "<b>Key Decisions:</b><ul>" + "".join(
                        f"<li><b>{d.get('decision','')}</b>: {d.get('rationale','')}</li>"
                        for d in arch["key_decisions"]
                    ) + "</ul>"
                validation = state.get("architecture_validation", {})
                if validation.get("verdict"):
                    vcolor = "#27AE60" if "approved" in validation["verdict"] else "#EB5757"
                    arch_text += (
                        f'<br><b>Validation:</b> '
                        f'<span style="color:{vcolor};font-weight:bold">'
                        f'{validation["verdict"].upper()}</span><br>'
                        f'{validation.get("validation_summary", "")}'
                    )
                if validation.get("flagged_issues"):
                    arch_text += "<br><b>Flagged:</b><ul>" + "".join(
                        f'<li style="color:#F2994A">[{i.get("severity","")}] '
                        f'{i.get("issue","")} — {i.get("recommendation","")}</li>'
                        for i in validation["flagged_issues"]
                    ) + "</ul>"

                files = list(state.get("generated_files", {}).keys())

                yield (
                    status_html,
                    run_btn_update,
                    stop_btn_update,
                    err_update,
                    _render_metrics(state),
                    _render_decisions(state.get("decision_audit", [])),
                    _render_tasks(state.get("task_plan", [])),
                    _render_tests(state.get("test_results", {})),
                    _render_security(state.get("security_findings", [])),
                    arch_text,
                    _render_preview(state),
                    _render_summary(state.get("workflow_summary", {})),
                    log_lines,
                    gr.update(choices=files),
                )

                if done:
                    break

        def view_file(filename):
            if not filename:
                return "", gr.update(interactive=False), ""
            state = get_state()
            content = state.get("generated_files", {}).get(filename, "")
            if not content:
                return f"# {filename} — not found", gr.update(interactive=False), ""
            # Return content as plain str — gr.Code/Textbox does not accept tuple
            return content, gr.update(interactive=True), ""

        def save_file(filename, new_content):
            if not filename:
                return gr.update(interactive=True), '<span style="color:#EB5757">⚠ No file selected</span>'
            state = get_state()
            state.setdefault("generated_files", {})[filename] = new_content
            set_state(state)
            from pathlib import Path
            out = Path("./generated_project") / filename
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(new_content)
                return gr.update(interactive=True), f'<span style="color:#27AE60">✅ Saved {filename}</span>'
            except Exception as e:
                return gr.update(interactive=True), f'<span style="color:#EB5757">Save error: {e}</span>'

        def ai_edit_file(filename, current_content, change_prompt):
            """Apply a natural-language edit to the selected file using the LLM."""
            if not filename:
                return current_content, '<span style="color:#EB5757">⚠ No file selected</span>'
            if not change_prompt.strip():
                return current_content, '<span style="color:#EB5757">⚠ Describe the change to make</span>'
            try:
                from core.llm import get_client
                client = get_client()
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are editing a source file. Apply ONLY the requested change. "
                            "Return the complete updated file as raw code. "
                            "No markdown fences, no explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"File: {filename}\n\n"
                            f"Current content:\n{current_content[:4000]}\n\n"
                            f"Change to apply: {change_prompt}\n\n"
                            "Return the complete updated file."
                        ),
                    },
                ]
                response, _ = client.call(messages, agent_name="AIEditor")
                new_code = response.strip()
                if new_code.startswith("```"):
                    lines = new_code.split("\n")
                    end = len(lines) - 1
                    while end > 0 and lines[end].strip().startswith("```"):
                        end -= 1
                    new_code = "\n".join(lines[1:end + 1]).strip()
                from pathlib import Path
                out = Path("./generated_project") / filename
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(new_code)
                state = get_state()
                state.setdefault("generated_files", {})[filename] = new_code
                set_state(state)
                return new_code, f'<span style="color:#27AE60">✅ Applied: {change_prompt[:60]}</span>'
            except Exception as e:
                return current_content, f'<span style="color:#EB5757">Error: {e}</span>'

        def download_zip():
            zip_path = _make_zip()
            if not zip_path:
                return gr.update(visible=False)
            return gr.update(value=zip_path, visible=True)

        file_selector.change(
            fn=view_file,
            inputs=[file_selector],
            outputs=[file_content, save_btn, save_status],
        )
        save_btn.click(
            fn=save_file,
            inputs=[file_selector, file_content],
            outputs=[save_btn, save_status],
        )
        ai_edit_btn.click(
            fn=ai_edit_file,
            inputs=[file_selector, file_content, edit_prompt],
            outputs=[file_content, ai_edit_status],
        )
        download_btn.click(
            fn=download_zip,
            inputs=[],
            outputs=[download_file],
        )

        def _classify_crash(err: str) -> str:
            """Return a human-friendly explanation of a crash cause."""
            e = err.lower()
            if "rate_limit" in e or "429" in e:
                return ("🚦 Groq rate limit hit. The pipeline made too many API calls too quickly. "
                        "Wait 60 seconds and try again, or reduce the number of tasks in your project.")
            if "token" in e and ("limit" in e or "exceed" in e or "context" in e):
                return ("📏 Token limit exceeded. The project was too large for one run. "
                        "Try a simpler project description, or break it into smaller scopes.")
            if "groq_api_key" in e or "api_key" in e or "authentication" in e:
                return ("🔑 Groq API key missing or invalid. "
                        "Make sure GROQ_API_KEY is set correctly in your environment.")
            if "connection" in e or "timeout" in e or "network" in e:
                return ("🌐 Network error connecting to Groq API. "
                        "Check your internet connection and try again.")
            if "json" in e and "parse" in e:
                return ("🧩 The LLM returned malformed JSON. This sometimes happens with complex prompts. "
                        "Try running again — it usually succeeds on the second attempt.")
            if "filenotfounderror" in e or "no such file" in e:
                return ("📂 A required file was not found. "
                        "Make sure the forge/ directory structure is intact.")
            return ("⚠️ An unexpected error stopped the pipeline. "
                    "See the technical details below, and check the Activity Log for context.")

        run_btn.click(
            fn=run_pipeline,
            inputs=[prompt_input, w_speed, w_quality, w_tests, w_security, w_simple],
            outputs=[
                status_badge,
                run_btn,
                stop_btn,
                error_banner,
                metrics_html,
                decisions_html,
                tasks_html,
                tests_html,
                security_html,
                arch_html,
                preview_html,
                summary_html,
                activity_log,
                file_selector,
            ],
        )

    return demo


def launch(share: bool = True, port: int = 7860):
    """Launch the Forge dashboard."""
    demo = create_dashboard()
    demo.launch(
        share=share,
        server_port=port,
        show_error=True,
        quiet=False,
    )