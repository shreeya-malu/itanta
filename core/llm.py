"""
core/llm.py
Groq client wrapper — unified interface for all agents.
Token-budget-aware: tracks spend and warns before hitting limits.
"""
from __future__ import annotations
import os, time, json, re
from typing import Any, Optional
from groq import Groq


def _load_config() -> dict:
    import yaml, pathlib
    cfg_path = pathlib.Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


CFG = _load_config()
LLM_CFG = CFG["llm"]

# Groq free-tier limit per minute for 70b model is 6000 tokens/min,
# but the per-request context window is 128k. We cap prompts to keep well under.
TOKEN_BUDGET_WARN = 80_000   # warn in logs after this many total tokens
TOKEN_BUDGET_HARD = 95_000   # switch to fast model after this (safety valve)


def _trim_messages(messages: list[dict], max_chars: int = 6000) -> list[dict]:
    """
    Smart-trim messages for token budget.

    CRITICAL: For codegen prompts, we must NOT blindly truncate from the end —
    that destroys import rules, must_define/must_not_define sections, and endpoint
    specs, which are the most important parts. Instead we:
      1. Always keep the system prompt intact.
      2. For codegen user prompts (detected by presence of key headers), we
         compress the MIDDLE (existing file content) and preserve the contract
         sections at the top and bottom.
      3. For all other messages, trim from the end as before.

    max_chars here is a soft guide — codegen prompts get up to 3x this budget
    because they carry necessary structural context.
    """
    trimmed = []
    for msg in messages:
        if msg["role"] == "system":
            # System prompts are always kept intact — they're already concise
            trimmed.append(msg)
            continue

        content = msg.get("content", "")
        # Codegen prompts can be large but must stay coherent — use a generous cap
        # (≈ 20k chars ≈ 5k tokens, well within Groq's 128k context window)
        codegen_max = max_chars * 4  # 20k chars for codegen
        if len(content) <= codegen_max:
            trimmed.append({**msg, "content": content})
            continue

        # Content is too long — apply smart compression
        # Detect if this is a codegen prompt (has structured sections)
        is_codegen = any(
            marker in content
            for marker in ("IMPORT RULES", "MUST DEFINE", "DO NOT DEFINE",
                           "ENDPOINTS TO IMPLEMENT", "EXTEND this existing file")
        )

        if is_codegen:
            # For codegen prompts: find the "Existing file:" block and compress it
            # Everything else (contract sections) is kept verbatim
            existing_marker = "\nExisting file:\n"
            if existing_marker in content:
                before, _, after = content.partition(existing_marker)
                # Find where the existing file block ends (next double-newline section)
                end_marker = "\n\nNew task to add:"
                if end_marker in after:
                    file_content, rest = after.split(end_marker, 1)
                    # Keep first 800 + last 400 chars of existing file to preserve structure
                    if len(file_content) > 1200:
                        file_content = (
                            file_content[:800]
                            + f"\n\n[... {len(file_content)-1200} chars omitted for context budget ...]\n\n"
                            + file_content[-400:]
                        )
                    content = before + existing_marker + file_content + end_marker + rest
                else:
                    # Can't find end marker — compress the after block
                    if len(after) > 2000:
                        after = after[:2000] + "\n\n[...existing file truncated...]"
                    content = before + existing_marker + after
            else:
                # No existing file block — keep first 75% + last 25% (preserve the contract tail)
                keep_head = int(codegen_max * 0.75)
                keep_tail = codegen_max - keep_head
                if len(content) > codegen_max:
                    content = (
                        content[:keep_head]
                        + f"\n\n[...{len(content) - codegen_max} chars omitted...]\n\n"
                        + content[-keep_tail:]
                    )
        else:
            # Non-codegen (planning, QA, etc.) — simple tail trim is fine
            content = content[:max_chars] + "\n\n[...truncated for token budget...]"

        trimmed.append({**msg, "content": content})
    return trimmed


class ForgeGroqClient:
    """Singleton Groq client shared across all agents."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Please set it via:\n"
                "  import os; os.environ['GROQ_API_KEY'] = 'gsk_...'"
            )
        self.client = Groq(api_key=api_key)
        self.total_tokens = 0
        self.total_calls = 0

    def call(
        self,
        messages: list[dict],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        agent_name: str = "unknown",
        expect_json: bool = False,
    ) -> tuple[str, dict]:
        """
        Make a Groq API call.
        Returns (response_text, usage_dict).
        Auto-trims prompts and falls back to fast model near token budget.
        """
        # Token budget safety valve — fall back to fast model if close to limit
        if self.total_tokens >= TOKEN_BUDGET_HARD and model == LLM_CFG["codegen_model"]:
            model = LLM_CFG["fast_model"]
            max_tokens = LLM_CFG.get("max_tokens_fast", 1024)

        model = model or LLM_CFG["codegen_model"]
        temperature = temperature if temperature is not None else LLM_CFG["temperature_codegen"]

        # Use smaller cap for fast model
        if model == LLM_CFG["fast_model"]:
            max_tokens = max_tokens or LLM_CFG.get("max_tokens_fast", 1024)
        else:
            max_tokens = max_tokens or LLM_CFG["max_tokens"]

        # Trim prompts to avoid blowing context — codegen gets a large budget
        messages = _trim_messages(messages, max_chars=5000)

        t0 = time.time()
        # httpx timeout: (connect_timeout, read_timeout)
        # connect_timeout catches stalled connections fast.
        # read_timeout is generous — a legitimate 2048-token response can take 60s+
        # on a loaded model. We never want to cut a response mid-generation.
        CONNECT_TIMEOUT = 15   # seconds to establish connection
        READ_TIMEOUT    = 180  # seconds to wait for full response (3 min max)

        def _do_call(mdl, msgs, temp, max_tok):
            import httpx
            return self.client.chat.completions.create(
                model=mdl,
                messages=msgs,
                temperature=temp,
                max_tokens=max_tok,
                timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
            )

        try:
            response = _do_call(model, messages, temperature, max_tokens)
        except Exception as e:
            err_str = str(e)
            # Rate limit — wait 20s and retry on fast model
            if "rate_limit" in err_str.lower() or "429" in err_str:
                log_msg = f"[{agent_name}] Rate limit hit — waiting 20s then retrying on fast model"
                print(log_msg)
                time.sleep(20)
                response = _do_call(
                    LLM_CFG["fast_model"], messages, temperature,
                    LLM_CFG.get("max_tokens_fast", 1024),
                )
            # Connection timeout (stalled connection, not slow generation) — retry once
            elif "connect" in err_str.lower() and "timeout" in err_str.lower():
                print(f"[{agent_name}] Connection timeout — retrying once")
                time.sleep(5)
                response = _do_call(model, messages, temperature, max_tokens)
            else:
                raise

        latency = time.time() - t0
        text = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency_s": round(latency, 3),
            "model": model,
            "agent": agent_name,
        }
        self.total_tokens += usage["total_tokens"]
        self.total_calls += 1

        if self.total_tokens >= TOKEN_BUDGET_WARN:
            print(f"[TokenBudget] WARNING: {self.total_tokens:,} tokens used — approaching limit")

        if expect_json:
            text = _extract_json(text)

        return text, usage

    def call_reasoning(self, messages: list[dict], agent_name: str = "unknown",
                       max_tokens: int = None) -> tuple[str, dict]:
        """Use the planning model (same as codegen, just lower temperature)."""
        return self.call(
            messages,
            model=LLM_CFG["planning_model"],
            temperature=LLM_CFG["temperature_reasoning"],
            max_tokens=max_tokens or LLM_CFG["max_tokens"],
            agent_name=agent_name,
        )

    def call_fast(self, messages: list[dict], agent_name: str = "unknown",
                  max_tokens: int = None) -> tuple[str, dict]:
        """Use the fast lightweight model — for boilerplate and non-Python files."""
        return self.call(
            messages,
            model=LLM_CFG["fast_model"],
            temperature=0.1,
            max_tokens=max_tokens or LLM_CFG.get("max_tokens_fast", 1024),
            agent_name=agent_name,
        )


def _extract_json(text: str) -> str:
    """Strip markdown fences and extract JSON from LLM output."""
    # Try to find JSON block
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()
    # Try to find raw JSON object/array
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_json_response(text: str) -> Any:
    """Parse JSON from LLM response, with fallback."""
    cleaned = _extract_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Second attempt: fix common LLM JSON mistakes
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse LLM JSON response: {e}\nRaw: {text[:500]}")


# Module-level singleton
_client: Optional[ForgeGroqClient] = None


def get_client() -> ForgeGroqClient:
    global _client
    if _client is None:
        _client = ForgeGroqClient()
    return _client


def reset_client():
    """Force re-initialisation (useful if API key is set after import)."""
    global _client
    _client = None