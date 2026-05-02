"""
agents/docker_agent.py

Template-based Docker file generation — no LLM calls.
The Dockerfile and docker-compose.yml are deterministic from the project stack.
This removes the hang risk entirely and produces more reliable output
than an LLM that might stall or return incomplete YAML.
"""
from __future__ import annotations
import time
from pathlib import Path
from core.observability import log_action, log_agent_metrics
from core.state import ForgeState


# ── Dockerfile template ────────────────────────────────────────────────────────

DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim

# Security: run as non-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Layer caching: install deps before copying code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')" || exit 1

CMD ["uvicorn", "{entrypoint}:app", "--host", "0.0.0.0", "--port", "{port}"]
"""


def _build_compose(port: int, db_url: str, entrypoint: str,
                   use_postgres: bool, use_mongo: bool, use_redis: bool) -> str:
    """
    Build docker-compose.yml programmatically using string concatenation
    instead of .format() — avoids YAML indentation bugs from template holes.
    """
    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "services:",
        "  app:",
        "    build: .",
        f"    ports:",
        f"      - \"{port}:{port}\"",
        "    environment:",
        f"      - DATABASE_URL={db_url}",
        "      - SECRET_KEY=changeme-in-production",
        "      - DEBUG=false",
    ]

    # depends_on
    deps = []
    if use_postgres: deps.append("db")
    if use_mongo:    deps.append("mongo")
    if use_redis:    deps.append("redis")

    if deps:
        lines.append("    depends_on:")
        for dep in deps:
            lines.append(f"      - {dep}")

    lines += [
        "    restart: unless-stopped",
        "    healthcheck:",
        "      test: [\"CMD\", \"python\", \"-c\",",
        f"             \"import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')\"]",
        "      interval: 30s",
        "      timeout: 10s",
        "      retries: 3",
    ]

    # ── External services ─────────────────────────────────────────────────────
    if use_postgres:
        lines += [
            "",
            "  db:",
            "    image: postgres:15-alpine",
            "    environment:",
            "      - POSTGRES_USER=forge",
            "      - POSTGRES_PASSWORD=forgepass",
            "      - POSTGRES_DB=forgedb",
            "    volumes:",
            "      - postgres_data:/var/lib/postgresql/data",
            "    restart: unless-stopped",
        ]

    if use_mongo:
        lines += [
            "",
            "  mongo:",
            "    image: mongo:7",
            "    environment:",
            "      - MONGO_INITDB_ROOT_USERNAME=forge",
            "      - MONGO_INITDB_ROOT_PASSWORD=forgepass",
            "    volumes:",
            "      - mongo_data:/data/db",
            "    restart: unless-stopped",
        ]

    if use_redis:
        lines += [
            "",
            "  redis:",
            "    image: redis:7-alpine",
            "    restart: unless-stopped",
        ]

    # ── Volumes ───────────────────────────────────────────────────────────────
    volume_names = []
    if use_postgres: volume_names.append("postgres_data")
    if use_mongo:    volume_names.append("mongo_data")

    if volume_names:
        lines += ["", "volumes:"]
        for v in volume_names:
            lines.append(f"  {v}:")

    return "\n".join(lines) + "\n"


def run(state: ForgeState) -> ForgeState:
    t0 = time.time()
    log_action("DockerAgent", "Generating Docker configuration (template-based)")

    inferred_stack  = state.get("inferred_stack", {})
    generated_files = state.get("generated_files", {})
    architecture    = state.get("architecture", {})

    # ── Detect stack ──────────────────────────────────────────────────────────
    stack_str = str(inferred_stack).lower()
    decisions = " ".join(
        d.get("decision", "") + " " + d.get("rationale", "")
        for d in architecture.get("key_decisions", [])
    ).lower()
    combined = stack_str + " " + decisions

    use_postgres = any(x in combined for x in ("postgres", "postgresql", "asyncpg"))
    use_mongo    = any(x in combined for x in ("mongo", "mongodb", "motor"))
    use_redis    = "redis" in combined

    # ── Detect entry point ────────────────────────────────────────────────────
    entrypoint = "app.main"
    for f in sorted(generated_files.keys()):
        if f.endswith("main.py"):
            entrypoint = f.replace("/", ".").removesuffix(".py")
            break

    port = 8000

    # ── Dockerfile ────────────────────────────────────────────────────────────
    dockerfile = DOCKERFILE_TEMPLATE.format(port=port, entrypoint=entrypoint)

    # ── docker-compose ────────────────────────────────────────────────────────
    if use_postgres:
        db_url = "postgresql://forge:forgepass@db:5432/forgedb"
    elif use_mongo:
        db_url = "mongodb://forge:forgepass@mongo:27017/forgedb"
    else:
        db_url = "sqlite:///./app.db"

    docker_compose = _build_compose(
        port=port,
        db_url=db_url,
        entrypoint=entrypoint,
        use_postgres=use_postgres,
        use_mongo=use_mongo,
        use_redis=use_redis,
    )

    # ── Write to disk ─────────────────────────────────────────────────────────
    project_dir = Path("./generated_project")
    project_dir.mkdir(exist_ok=True)

    for filename, content in [("Dockerfile", dockerfile), ("docker-compose.yml", docker_compose)]:
        generated_files[filename] = content
        (project_dir / filename).write_text(content)
        log_action("DockerAgent", f"  ✓ {filename}")

    stack_detected = "postgres" if use_postgres else "mongo" if use_mongo else "sqlite"
    metrics = {"latency_s": round(time.time() - t0, 3), "tokens": 0, "stack": stack_detected}
    log_agent_metrics("DockerAgent", metrics)

    state["dockerfile"]                   = dockerfile
    state["docker_compose"]               = docker_compose
    state["docker_validated"]             = True
    state["generated_files"]              = generated_files
    state["agent_metrics"]["DockerAgent"] = metrics
    state["phase"]                        = "docker_done"

    log_action("DockerAgent", "Docker configuration complete",
               f"stack={stack_detected} entry={entrypoint} — no LLM call")
    return state