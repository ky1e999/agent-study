"""
Compare ReAct alone vs combinations with CoT, ToT, GoT-style graphs.
Orchestration in Python; same mock tools as reasoning-demos.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()

SYSTEM_MINIMAL = "You are a helpful assistant."

# When True, every LLM request prints full system/user or multi-turn context first.
_TRACE_PROMPTS = True
_LLM_CALL_SEQ = 0


def configure_prompt_trace(enabled: bool) -> None:
    global _TRACE_PROMPTS
    _TRACE_PROMPTS = enabled


def reset_llm_call_counter() -> None:
    global _LLM_CALL_SEQ
    _LLM_CALL_SEQ = 0


def _next_llm_n() -> int:
    global _LLM_CALL_SEQ
    _LLM_CALL_SEQ += 1
    return _LLM_CALL_SEQ


def _print_llm_context(kind: str, label: str, body: str) -> None:
    if not _TRACE_PROMPTS:
        return
    n = _next_llm_n()
    print(
        f"\n{'─' * 72}\n▶ LLM call #{n}  [{kind}]  {label}\n{'─' * 72}\n{body.rstrip()}\n",
        flush=True,
    )


def _normalize_message_content(content: Any) -> str:
    """OpenAI-style content: str, or list of {type, text} parts → one string for display."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text" and "text" in p:
                    parts.append(str(p["text"]))
                elif "text" in p:
                    parts.append(str(p["text"]))
                else:
                    parts.append(json.dumps(p, ensure_ascii=False))
            else:
                txt = getattr(p, "text", None)
                parts.append(str(txt) if txt is not None else str(p))
        return "\n".join(parts)
    return str(content)


def _try_pretty_json(text: str, *, max_chars: int = 8000) -> str:
    """Pretty-print embedded JSON objects/arrays; keep trailing non-JSON lines as-is."""

    def shrink(s: str) -> str:
        return s[:max_chars] + "\n…" if len(s) > max_chars else s

    def try_dump(raw: str) -> str | None:
        try:
            obj = json.loads(raw)
            return shrink(json.dumps(obj, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            return None

    t = text.strip("\ufeff").strip()
    if len(t) < 2:
        return text

    # Whole string is one JSON value
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        got = try_dump(t)
        if got is not None:
            return got

    # Balanced {...} prefix (JSON then "\nExtra instructions...")
    if "{" in t:
        start = t.index("{")
        depth = 0
        for i, ch in enumerate(t[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = t[start : i + 1]
                    got = try_dump(candidate)
                    if got is not None:
                        rest = t[i + 1 :].strip()
                        if rest:
                            return got + "\n\n" + rest
                        return got
                    break

    return text


def _message_preview(m: Any) -> dict[str, Any]:
    if isinstance(m, dict):
        return dict(m)
    if hasattr(m, "model_dump"):
        return m.model_dump()
    out: dict[str, Any] = {"role": getattr(m, "role", None)}
    if hasattr(m, "content"):
        out["content"] = m.content
    if getattr(m, "tool_calls", None):
        out["tool_calls"] = m.tool_calls
    if getattr(m, "tool_call_id", None):
        out["tool_call_id"] = m.tool_call_id
    return out


def _format_messages_for_trace(
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    *,
    temperature: float,
    tools: list[dict[str, Any]] | None,
    trace_label: str,
) -> str:
    lines = [f"model: {model}", f"temperature: {temperature}", f"label: {trace_label}", ""]
    for i, raw in enumerate(messages):
        md = _message_preview(raw)
        role = md.get("role", "?")
        lines.append(f"--- message[{i}] role={role} ---")
        c = md.get("content")
        if c:
            lines.append(_try_pretty_json(_normalize_message_content(c)))
        tcs = md.get("tool_calls")
        if tcs:
            for tc in tcs:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    fname = fn.get("name", "") if isinstance(fn, dict) else ""
                    fargs = fn.get("arguments", "") if isinstance(fn, dict) else ""
                else:
                    fn = getattr(tc, "function", None)
                    fname = getattr(fn, "name", "") if fn else ""
                    fargs = getattr(fn, "arguments", "") if fn else ""
                arg_s = _try_pretty_json(str(fargs), max_chars=4000)
                if len(arg_s) > 1200 and "\n" not in arg_s:
                    arg_s = arg_s[:1200] + "…"
                lines.append(f"  → tool_call: {fname}\n{arg_s}")
        tid = md.get("tool_call_id")
        if tid:
            lines.append(f"  (responding to tool_call_id={tid})")
    if tools:
        names = [t.get("function", {}).get("name") for t in tools if t.get("type") == "function"]
        lines.append("")
        lines.append(f"[Tool schemas in this request: {names}]")
    return "\n".join(lines)


DEFAULT_QUESTION = (
    "A shop sells apples for $2 each and oranges for $3 each. "
    "Alice buys 4 apples and 2 oranges. Bob buys half as many apples as Alice "
    "and twice as many oranges. How much does Bob pay in dollars?"
)


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY in environment or agent-study/.env", file=sys.stderr)
        sys.exit(1)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    hdrs: dict[str, str] = {}
    if ref := os.environ.get("HTTP_REFERER"):
        hdrs["HTTP-Referer"] = ref
    if title := os.environ.get("X_TITLE"):
        hdrs["X-Title"] = title
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=hdrs or None)


def chat_text(
    client: OpenAI,
    model: str,
    user: str,
    *,
    temperature: float = 0.2,
    trace_label: str = "chat_text",
) -> str:
    if _TRACE_PROMPTS:
        user_disp = _try_pretty_json(user)
        body = (
            f"model: {model}\ntemperature: {temperature}\n\n"
            f"system:\n{SYSTEM_MINIMAL}\n\nuser:\n{user_disp}"
        )
        _print_llm_context("chat.completions (no tools)", trace_label, body)
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_MINIMAL},
            {"role": "user", "content": user},
        ],
    )
    content = (r.choices[0].message.content or "").strip()
    if _TRACE_PROMPTS:
        prev = _try_pretty_json(content, max_chars=12000)
        print(f"◀ assistant reply:\n{prev}\n{'·' * 72}\n", flush=True)
    return content


def chat_completion(
    client: OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    *,
    temperature: float = 0.2,
    tools: list[dict[str, Any]] | None = None,
    trace_label: str = "chat_completion",
):
    if _TRACE_PROMPTS:
        body = _format_messages_for_trace(
            model, messages, temperature=temperature, tools=tools, trace_label=trace_label
        )
        kind = "chat.completions + tools" if tools else "chat.completions"
        _print_llm_context(kind, trace_label, body)
    kw: dict[str, Any] = {"model": model, "messages": list(messages), "temperature": temperature}
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
    r = client.chat.completions.create(**kw)
    if _TRACE_PROMPTS:
        msg = r.choices[0].message
        c_raw = (msg.content or "").strip()
        pretty_c = _try_pretty_json(c_raw, max_chars=12000) if c_raw else "(empty)"
        parts_reply = [f"content:\n{pretty_c}"]
        if msg.tool_calls:
            for tc in msg.tool_calls:
                pa = _try_pretty_json(str(tc.function.arguments), max_chars=4000)
                parts_reply.append(f"tool_call: {tc.function.name}\n{pa}")
        print(f"◀ assistant reply:\n" + "\n".join(parts_reply) + f"\n{'·' * 72}\n", flush=True)
    return r


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    d = json.loads(text)
    if not isinstance(d, dict):
        raise ValueError("expected object")
    return d


REACT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Retrieve a short text snippet for a query (mock catalog).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate a simple arithmetic expression (digits, + - * / . parens).",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final answer to the user task.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


def mock_search(query: str) -> str:
    q = query.lower()
    if "price" in q or "apple" in q or "orange" in q:
        return "Catalog: apples $2 each, oranges $3 each."
    return "No extra facts found."


def mock_calc(expression: str) -> str:
    allowed = set("0123456789+-*/. ()")
    if not all(c in allowed for c in expression):
        return "Error: only arithmetic characters allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def dispatch_react_tool(name: str, arguments: str) -> str:
    args = json.loads(arguments or "{}")
    if name == "search":
        return mock_search(str(args.get("query", "")))
    if name == "calc":
        return mock_calc(str(args.get("expression", "")))
    if name == "finish":
        return "__FINISH__:" + str(args.get("answer", ""))
    return "Unknown tool"


def run_react_tool_loop(
    client: OpenAI,
    model: str,
    seed_messages: list[ChatCompletionMessageParam],
    *,
    max_rounds: int = 8,
    log_prefix: str = "r",
) -> tuple[str | None, list[str]]:
    messages: list[ChatCompletionMessageParam] = list(seed_messages)
    logs: list[str] = []
    final: str | None = None
    for i in range(max_rounds):
        resp = chat_completion(
            client,
            model,
            messages,
            temperature=0.1,
            tools=REACT_TOOLS,
            trace_label=f"ReAct/{log_prefix} round {i + 1}",
        )
        msg = resp.choices[0].message
        tcalls = msg.tool_calls or []
        am: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if tcalls:
            am["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tcalls
            ]
        messages.append(am)  # type: ignore[arg-type]
        logs.append(f"{log_prefix}-{i + 1} assistant tool_calls={len(tcalls)} content={msg.content!r}")
        if not tcalls:
            break
        for tc in tcalls:
            out = dispatch_react_tool(tc.function.name, tc.function.arguments)
            if out.startswith("__FINISH__:"):
                final = out.removeprefix("__FINISH__:")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Recorded."})
                return final, logs
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            logs.append(f"  tool {tc.function.name} -> {out!r}")
    return final, logs


# --- ToT shallow helpers ---


def _tot_hypothesis_user(question: str, branch_id: int) -> str:
    return (
        json.dumps({"task": question, "branch_id": branch_id}, ensure_ascii=False)
        + '\nReturn ONLY JSON: {"hypothesis": string} — one line plan, no final numeric answer yet.'
    )


def _tot_score_user(question: str, hypotheses: list[str]) -> str:
    return (
        json.dumps({"task": question, "candidates": hypotheses}, ensure_ascii=False)
        + '\nReturn ONLY JSON: {"scores": [number, number, number], "best_index": 0|1|2}.'
    )


def pick_tot_hypothesis(client: OpenAI, model: str, question: str) -> tuple[str, list[str]]:
    log: list[str] = []
    hypotheses: list[str] = []
    for k in range(3):
        raw = chat_text(
            client,
            model,
            _tot_hypothesis_user(question, k),
            temperature=0.45 + 0.1 * k,
            trace_label=f"ToT hypothesis k={k}",
        )
        try:
            hypotheses.append(str(parse_json_object(raw).get("hypothesis", "")))
        except (json.JSONDecodeError, ValueError):
            hypotheses.append(raw[:200])
        log.append(f"hyp{k}: {raw[:300]}")
    score_raw = chat_text(
        client,
        model,
        _tot_score_user(question, hypotheses),
        temperature=0.0,
        trace_label="ToT score 3 hypotheses",
    )
    log.append(f"score: {score_raw[:500]}")
    best_i = 0
    try:
        sc = parse_json_object(score_raw)
        bi = sc.get("best_index", 0)
        if isinstance(bi, int) and 0 <= bi < len(hypotheses):
            best_i = bi
    except (json.JSONDecodeError, ValueError):
        pass
    return hypotheses[best_i] if hypotheses else "", log


# --- Combos ---


def combo_react_only(client: OpenAI, model: str, question: str) -> str:
    seed: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_MINIMAL},
        {"role": "user", "content": question},
    ]
    final, logs = run_react_tool_loop(client, model, seed, max_rounds=12)
    return (final or "(no finish)") + "\n---logs---\n" + "\n".join(logs[-15:])


def combo_cot_then_react(client: OpenAI, model: str, question: str) -> str:
    """Structured CoT plan (JSON, no tools), then ReAct with that context."""
    u = (
        json.dumps({"task": question}, ensure_ascii=False)
        + '\nReturn ONLY JSON: {"reasoning_steps": [string]} — 3–7 short steps; '
        "no tool use; do not give the final number until tools verified later."
    )
    raw = chat_text(
        client, model, u, temperature=0.25, trace_label="CoT→ReAct: JSON reasoning_steps plan"
    )
    try:
        steps = parse_json_object(raw).get("reasoning_steps") or []
    except (json.JSONDecodeError, ValueError):
        steps = [raw[:400]]
    steps = [str(s) for s in steps if s]
    seed_u = (
        json.dumps({"task": question, "reasoning_steps": steps}, ensure_ascii=False)
        + "\nUse search/calc as needed. Call finish with the final answer."
    )
    seed: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_MINIMAL},
        {"role": "user", "content": seed_u},
    ]
    final, logs = run_react_tool_loop(client, model, seed, max_rounds=12)
    body = (
        f"plan_steps={steps!r}\nfinal={(final or '(no finish)')!r}\n---logs---\n"
        + "\n".join(logs[-20:])
    )
    return body


def combo_cot_interleaved_react(client: OpenAI, model: str, question: str) -> str:
    """Alternate: one JSON reasoning step, then short ReAct burst; feed traces forward."""
    cot: list[str] = []
    obs_trace: list[str] = []
    final: str | None = None
    all_logs: list[str] = []
    for cycle in range(6):
        cu = json.dumps(
            {
                "task": question,
                "reasoning_so_far": cot,
                "recent_tool_activity": obs_trace[-3:],
            },
            ensure_ascii=False,
        ) + (
            '\nReturn ONLY JSON: {"reasoning_step": string} — one next inference or plan tweak; '
            'no tool use in this reply.'
        )
        cr = chat_text(
            client,
            model,
            cu,
            temperature=0.2,
            trace_label=f"interleaved: CoT step cycle {cycle}",
        )
        try:
            step = str(parse_json_object(cr).get("reasoning_step", "")).strip()
        except (json.JSONDecodeError, ValueError):
            step = cr[:180]
        if step:
            cot.append(step)
        su = (
            json.dumps({"task": question, "reasoning_chain": cot}, ensure_ascii=False)
            + "\nUse search/calc to verify or compute. Call finish when you have the full answer."
        )
        seed: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_MINIMAL},
            {"role": "user", "content": su},
        ]
        final, logs = run_react_tool_loop(client, model, seed, max_rounds=4, log_prefix=f"c{cycle}")
        all_logs.extend(logs)
        obs_trace.extend(logs)
        if final:
            break
    return (
        f"cot_steps={cot!r}\nfinal={(final or '(no finish)')!r}\n---logs---\n"
        + "\n".join(all_logs[-25:])
    )


def combo_tot_cot_then_react(client: OpenAI, model: str, question: str) -> str:
    """ToT (pick strategy) → CoT elaborate under that strategy → ReAct."""
    chosen, tlog = pick_tot_hypothesis(client, model, question)
    u = (
        json.dumps({"task": question, "strategy": chosen}, ensure_ascii=False)
        + '\nReturn ONLY JSON: {"reasoning_steps": [string]} — 2–5 steps following this strategy only; no tools.'
    )
    raw = chat_text(
        client,
        model,
        u,
        temperature=0.2,
        trace_label="ToT+CoT: steps under chosen strategy",
    )
    try:
        steps = parse_json_object(raw).get("reasoning_steps") or []
    except (json.JSONDecodeError, ValueError):
        steps = [raw[:300]]
    steps = [str(s) for s in steps if s]
    plan = [chosen, *steps]
    seed_u = (
        json.dumps({"task": question, "tot_strategy": chosen, "reasoning_steps": steps}, ensure_ascii=False)
        + "\nUse search/calc. finish with the task answer."
    )
    seed: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_MINIMAL},
        {"role": "user", "content": seed_u},
    ]
    final, rlogs = run_react_tool_loop(client, model, seed, max_rounds=12)
    return (
        "\n".join(tlog)
        + f"\nplan={plan!r}\nfinal={(final or '(no finish)')!r}\n---react---\n"
        + "\n".join(rlogs[-20:])
    )


def combo_tot_then_react(client: OpenAI, model: str, question: str) -> str:
    chosen, tlog = pick_tot_hypothesis(client, model, question)
    seed_u = (
        json.dumps({"task": question, "chosen_approach": chosen}, ensure_ascii=False)
        + "\nExecute with search/calc. finish(answer) when done."
    )
    seed: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_MINIMAL},
        {"role": "user", "content": seed_u},
    ]
    final, rlogs = run_react_tool_loop(client, model, seed, max_rounds=12)
    return (
        "\n".join(tlog)
        + f"\nchosen={chosen!r}\nfinal={(final or '(no finish)')!r}\n---react---\n"
        + "\n".join(rlogs[-20:])
    )


def combo_tot_react_replan(client: OpenAI, model: str, question: str) -> str:
    chosen, tlog = pick_tot_hypothesis(client, model, question)
    parts = list(tlog)

    def run_r(ch: str) -> tuple[str | None, list[str]]:
        su = (
            json.dumps({"task": question, "chosen_approach": ch}, ensure_ascii=False)
            + "\nUse search/calc. finish(answer) with the task answer."
        )
        seed: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_MINIMAL},
            {"role": "user", "content": su},
        ]
        return run_react_tool_loop(client, model, seed, max_rounds=8)

    final, rlogs = run_r(chosen)
    parts.append(f"react1 final={final!r}")
    parts.extend(rlogs[-12:])

    if final:
        return "\n".join(parts)

    fail_ctx = "\n".join(rlogs[-8:])[:1200]
    u2 = (
        json.dumps(
            {
                "task": question,
                "failed_approach": chosen,
                "failure_evidence": fail_ctx,
            },
            ensure_ascii=False,
        )
        + '\nReturn ONLY JSON: {"hypotheses": [string, string, string]} — three new approaches.'
    )
    raw2 = chat_text(
        client,
        model,
        u2,
        temperature=0.35,
        trace_label="replan: 3 new hypotheses after ReAct failure",
    )
    try:
        hyps = parse_json_object(raw2).get("hypotheses") or []
        hyps = [str(h) for h in hyps[:3]]
    except (json.JSONDecodeError, ValueError):
        hyps = [raw2[:200]]
    if len(hyps) < 3:
        hyps = (hyps * 3)[:3]
    score2 = chat_text(
        client,
        model,
        _tot_score_user(question, hyps),
        temperature=0.0,
        trace_label="replan: score new hypotheses",
    )
    bi = 0
    try:
        bi = int(parse_json_object(score2).get("best_index", 0))
    except (json.JSONDecodeError, ValueError, TypeError):
        bi = 0
    bi = max(0, min(bi, len(hyps) - 1))
    chosen2 = hyps[bi]
    parts.append(f"replan pick: {chosen2!r}")
    final2, r2 = run_r(chosen2)
    parts.append(f"react2 final={final2!r}")
    parts.extend(r2[-15:])
    return "\n".join(parts)


def _topo_node_ids(nodes: list[dict[str, Any]]) -> list[str]:
    ids = [str(n["id"]) for n in nodes if n.get("id")]
    deps_map: dict[str, set[str]] = {}
    id_set = set(ids)
    for n in nodes:
        nid = str(n.get("id", ""))
        raw = n.get("deps") or []
        deps_map[nid] = {str(d) for d in raw if str(d) in id_set}
    ready = [i for i in ids if not deps_map.get(i, set())]
    out: list[str] = []
    pending = set(ids)
    while ready:
        n = ready.pop(0)
        if n not in pending:
            continue
        out.append(n)
        pending.remove(n)
        for m in ids:
            if n in deps_map.get(m, set()):
                deps_map[m].discard(n)
                if not deps_map[m] and m in pending and m not in ready:
                    ready.append(m)
    if len(out) != len(ids):
        return ids
    return out


def combo_got_react(client: OpenAI, model: str, question: str) -> str:
    gu = (
        json.dumps({"task": question}, ensure_ascii=False)
        + '\nReturn ONLY JSON: {"nodes": [{"id": string, "goal": string, "deps": [string]}]} — '
        "3–5 nodes; deps lists prerequisite ids; cover: get prices / quantities / bob total."
    )
    raw = chat_text(
        client,
        model,
        gu,
        temperature=0.25,
        trace_label="GoT: emit node DAG JSON",
    )
    try:
        nodes = parse_json_object(raw).get("nodes") or []
    except (json.JSONDecodeError, ValueError):
        nodes = []
    if not nodes:
        return f"(no graph parsed)\n{raw[:500]}"
    order = _topo_node_ids(list(nodes))
    logs: list[str] = [f"topo_order={order}"]
    completed: dict[str, str] = {}

    for nid in order:
        node = next((n for n in nodes if str(n.get("id")) == nid), None)
        if not node:
            continue
        goal = str(node.get("goal", ""))
        dep_out = {d: completed.get(d, "") for d in (node.get("deps") or [])}
        nu = (
            json.dumps(
                {
                    "task": question,
                    "subgoal_id": nid,
                    "subgoal": goal,
                    "prerequisite_outputs": dep_out,
                },
                ensure_ascii=False,
            )
            + "\nUse tools for this subgoal only. finish(answer) with a short outcome "
            "(number or fact), not the full original task unless this subgoal is the final total."
        )
        seed: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_MINIMAL},
            {"role": "user", "content": nu},
        ]
        fn, lg = run_react_tool_loop(client, model, seed, max_rounds=6, log_prefix=f"n.{nid}")
        completed[nid] = fn or "(no sub finish)"
        logs.append(f"node {nid}: {completed[nid]!r}")
        logs.extend(lg[-5:])

    mu = (
        json.dumps({"task": question, "node_results": completed}, ensure_ascii=False)
        + '\nReturn ONLY JSON: {"final_answer": string}.'
    )
    merged = chat_text(
        client, model, mu, temperature=0.0, trace_label="GoT: merge node_results → final_answer"
    )
    logs.append(f"merge: {merged[:600]}")
    return "\n".join(logs)


COMBOS: dict[str, Any] = {
    "react_only": combo_react_only,
    "cot_then_react": combo_cot_then_react,
    "cot_interleaved_react": combo_cot_interleaved_react,
    "tot_then_react": combo_tot_then_react,
    "tot_cot_then_react": combo_tot_cot_then_react,
    "tot_react_replan": combo_tot_react_replan,
    "got_react": combo_got_react,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--combo",
        choices=[*COMBOS.keys(), "all"],
        default="all",
    )
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument(
        "--quiet-prompts",
        action="store_true",
        help="Do not print LLM request/response traces (recommended with --combo all).",
    )
    args = ap.parse_args()

    configure_prompt_trace(not args.quiet_prompts)

    client = build_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def run(name: str) -> tuple[str, float]:
        reset_llm_call_counter()
        if _TRACE_PROMPTS:
            print(f"\n{'█' * 72}\nCOMBO: {name}\n{'█' * 72}\n", flush=True)
        fn = COMBOS[name]
        t0 = time.perf_counter()
        out = fn(client, model, args.question)
        return out, time.perf_counter() - t0

    if args.combo == "all":
        for name in COMBOS:
            out, dt = run(name)
            preview = out.replace("\n", " ")[:140] + ("…" if len(out) > 140 else "")
            print(f"\n[{name}] {dt:.2f}s  preview: {preview}")
    else:
        out, dt = run(args.combo)
        print(f"\n[{args.combo}] {dt:.2f}s")
        print("\n" + "=" * 72)
        print(out)


if __name__ == "__main__":
    main()
