"""Deterministic plain-text rendering of canonical coding transcripts, and grader contexts.

Rules implemented (docs/data_schema.md, "Canonical transcript and rendering"):
- the system prompt is never part of the steps, so it is never rendered;
- each step is rendered in order with "=== USER (task) ===", "=== ASSISTANT ===",
  "--- TOOL CALL: name ---", "--- TOOL OUTPUT ---" headers (environment-injected user
  messages get "=== USER (environment) ===");
- execute_bash input is the raw command; str_replace_editor is "command path" plus its text
  fields raw and labelled (never json.dumps, which would escape newlines);
- any single tool output over N tokens keeps its first N/2 and last N/2 tokens around
  "[... truncated K tokens ...]";
- the final `finish` step is not rendered by default: its content and message are the
  agent's final report, which graders receive separately as the variant.

Token counts use tiktoken o200k_base, a stand-in for the graders' own tokenizers.
Everything here treats trajectory text as plain strings; nothing is executed.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

ENCODING_NAME = "o200k_base"
TRUNCATION_MARKER = "\n[... truncated {removed} tokens ...]\n"
NO_OUTPUT = "[no output]"

# str_replace_editor fields rendered after "command path", in this order, each labelled.
_EDITOR_FIELDS = ("old_str", "new_str", "file_text", "insert_line", "view_range")


@lru_cache(maxsize=1)
def get_encoder() -> Any:
    """tiktoken.get_encoding("o200k_base"), loaded once.

    The BPE file is cached locally after the first download (TIKTOKEN_CACHE_DIR or tiktoken's
    default cache). If loading fails, the error says so instead of silently falling back.
    """
    import tiktoken

    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception as exc:  # network or cache problem
        raise RuntimeError(
            f"could not load tiktoken encoding {ENCODING_NAME!r} ({exc}); set TIKTOKEN_CACHE_DIR to a "
            "directory that holds the cached BPE file"
        ) from exc


def _encode(text: str) -> list[int]:
    # disallowed_special=(): special-token strings inside trajectories are ordinary text.
    return get_encoder().encode(text, disallowed_special=())


def count_tokens(text: str) -> int:
    """Number of o200k_base tokens in text (special-token strings counted as plain text)."""
    if not text:
        return 0
    return len(_encode(text))


def _decode_prefix(tokens: list[int], n: int) -> str:
    """Decode tokens[:n], moving the cut back (at most 3 tokens) to a UTF-8 character boundary.

    If no clean boundary is found, decode with errors="replace" (a U+FFFD may appear).
    """
    enc = get_encoder()
    for k in range(n, max(n - 4, 0) - 1, -1):
        try:
            return enc.decode_bytes(tokens[:k]).decode("utf-8")
        except UnicodeDecodeError:
            continue
    return enc.decode(tokens[:n], errors="replace")


def _decode_suffix(tokens: list[int], n: int) -> str:
    """Decode the last n tokens, moving the cut forward (at most 3 tokens) to a character boundary."""
    enc = get_encoder()
    start = len(tokens) - n
    for s in range(start, min(start + 4, len(tokens)) + 1):
        try:
            return enc.decode_bytes(tokens[s:]).decode("utf-8")
        except UnicodeDecodeError:
            continue
    return enc.decode(tokens[start:], errors="replace")


def truncate_middle(text: str, max_tokens: int = 2000, head: int = 1000, tail: int = 1000) -> tuple[str, int]:
    """Keep the first `head` and last `tail` tokens of a text longer than `max_tokens`.

    Returns (text, removed_tokens). If count_tokens(text) <= max_tokens: (text, 0). Otherwise
    head + "\\n[... truncated {removed} tokens ...]\\n" + tail, where removed is the number of
    tokens dropped from the middle. Cuts are moved to UTF-8 character boundaries when a token
    boundary falls inside a multi-byte character (errors="replace" as the last resort).
    """
    if len(text) <= max_tokens:  # a token covers at least one character
        return text, 0
    tokens = _encode(text)
    if len(tokens) <= max_tokens:
        return text, 0
    removed = len(tokens) - head - tail
    return _decode_prefix(tokens, head) + TRUNCATION_MARKER.format(removed=removed) + _decode_suffix(tokens, tail), removed


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def render_tool_input(tool_name: str | None, tool_input: dict | None, tool_input_raw: str | None = None) -> str:
    """Render one tool call's arguments as readable text (see the module docstring)."""
    if tool_input is None:
        return tool_input_raw if tool_input_raw is not None else ""
    if tool_name == "execute_bash":
        return str(tool_input.get("command", ""))
    if tool_name == "str_replace_editor":
        head = f"{tool_input.get('command', '')} {tool_input.get('path', '')}".rstrip()
        parts = [head]
        for key in _EDITOR_FIELDS:
            if key in tool_input and tool_input[key] is not None:
                value = tool_input[key]
                text = value if isinstance(value, str) else _compact_json(value)
                parts.append(f"{key}:\n{text}")
        return "\n".join(parts)
    if tool_name == "think":
        return str(tool_input.get("thought", ""))
    if tool_name == "finish":
        return str(tool_input.get("message", ""))
    if tool_name == "task_tracker":
        return _compact_json(tool_input)
    return json.dumps(tool_input, sort_keys=True, ensure_ascii=False)


def render_step(step: dict, truncate_tool_outputs_over: int | None) -> str:
    """Render one canonical step. Tool outputs over the limit are middle-truncated."""
    if step.get("role") == "user":
        label = "environment" if step.get("source") == "environment" else "task"
        return f"=== USER ({label}) ===\n{step.get('content', '')}\n"
    text = f"=== ASSISTANT ===\n{step.get('content') or ''}\n"
    tool_name = step.get("tool_name")
    if tool_name:
        rendered_input = render_tool_input(tool_name, step.get("tool_input"), step.get("tool_input_raw"))
        output = step.get("tool_output")
        if output is None:
            output = NO_OUTPUT
        elif truncate_tool_outputs_over is not None:
            half = truncate_tool_outputs_over // 2
            output, _ = truncate_middle(output, truncate_tool_outputs_over, half, truncate_tool_outputs_over - half)
        text += f"--- TOOL CALL: {tool_name} ---\n{rendered_input}\n--- TOOL OUTPUT ---\n{output}\n"
    return text


def render_transcript(steps: list[dict], truncate_tool_outputs_over: int | None = 2000, include_finish: bool = False) -> str:
    """Render all steps in order, separated by a blank line.

    The final step is skipped when its tool_name is "finish" and include_finish is False.
    Same input -> identical string.
    """
    if not include_finish and steps and steps[-1].get("tool_name") == "finish":
        steps = steps[:-1]
    return "\n".join(render_step(step, truncate_tool_outputs_over) for step in steps)


def hidden_context(repo: str, issue_text: str) -> str:
    """The grader's view in the hidden condition: the task only."""
    return f"Fix the issue described below, then summarize what you did.\n\nRepository: {repo}\n\nIssue:\n{issue_text}"


def shown_context(repo: str, issue_text: str, transcript_text: str) -> str:
    """The grader's view in the shown condition: the task plus the rendered transcript."""
    return (
        hidden_context(repo, issue_text)
        + "\n\nAgent transcript (any single tool output over 2,000 tokens is truncated and marked):\n"
        + transcript_text
    )
