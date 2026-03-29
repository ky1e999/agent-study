"""
Reasoning demos where the *pattern* is implemented in Python:
JSON step loops (CoT), native tool loops (ReAct), structured plan then per-step tools (Plan-and-Act),
tool+executor (PoT), tree search in code (ToT: beam-BFS / DFS + shallow baseline), vote over chains (CoT-SC), explicit memory store + compaction (MoR).

The model sees short, mechanical user messages (schemas / tool defs), not
"you are now in ReAct mode" style prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

# Generic system line only; pattern lives in user payloads + loops + tools.
SYSTEM_MINIMAL = "You are a helpful assistant."

DEFAULT_QUESTION = (
    "A shop sells apples for $2 each and oranges for $3 each. "
    "Alice buys 4 apples and 2 oranges. Bob buys half as many apples as Alice "
    "and twice as many oranges. How much does Bob pay in dollars?"
)


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY in .env (see env.example).", file=sys.stderr)
        sys.exit(1)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_headers: dict[str, str] = {}
    if ref := os.environ.get("HTTP_REFERER"):
        default_headers["HTTP-Referer"] = ref
    if title := os.environ.get("X_TITLE"):
        default_headers["X-Title"] = title
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers or None)


def chat_text(
    client: OpenAI,
    model: str,
    user: str,
    *,
    system: str = SYSTEM_MINIMAL,
    temperature: float = 0.2,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    msg = resp.choices[0].message
    return (msg.content or "").strip()


def chat_completion(
    client: OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    *,
    temperature: float = 0.2,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = "auto",
):
    kwargs: dict[str, Any] = dict(model=model, messages=list(messages), temperature=temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError("expected JSON object")
    return decoded


def vote_normalize(answer: str) -> str:
    s = answer.strip()
    s = re.sub(r"(?i)\s*(usd|dollars?)\s*$", "", s).strip()
    s = s.replace("$", "").replace(",", "").strip().rstrip(".")
    if re.fullmatch(r"-?\d+\.?\d*", s):
        try:
            x = float(s)
            return str(int(x)) if x == int(x) else str(x)
        except ValueError:
            pass
    return s.lower()


def print_section(title: str, body: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(body.strip())


# --- CoT: code-driven multi-step JSON loop (no “chain-of-thought” phrasing) ---


def _cot_user_payload(question: str, prior_steps: list[str]) -> str:
    spec = (
        "Extend the solution by one step. Reply with ONLY a JSON object (no markdown) with keys: "
        '"step" (string: one short inference or calculation for this turn; use "" if already finished), '
        '"done" (boolean), '
        '"answer" (string or null; the final answer when done is true).'
    )
    return json.dumps({"task": question, "prior_steps": prior_steps}, ensure_ascii=False) + "\n" + spec


def run_json_step_chain(
    client: OpenAI,
    model: str,
    question: str,
    *,
    max_steps: int = 12,
    temperature: float = 0.2,
) -> tuple[str, list[str]]:
    prior: list[str] = []
    traces: list[str] = []
    for _ in range(max_steps):
        user = _cot_user_payload(question, prior)
        raw = chat_text(client, model, user, temperature=temperature)
        traces.append(raw)
        try:
            data = parse_json_object(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        step = (data.get("step") or "").strip()
        done = bool(data.get("done"))
        ans = data.get("answer")
        if step:
            prior.append(step)
        if done and ans is not None and str(ans).strip():
            return str(ans).strip(), traces
    return "(no conclusion)", traces


def demo_cot(client: OpenAI, model: str, question: str) -> str:
    final, traces = run_json_step_chain(client, model, question)
    body = (
        f"Final (from code loop): {final!r}\n\n"
        + "--- raw model outputs per turn ---\n"
        + "\n---\n".join(f"turn {i + 1}:\n{t}" for i, t in enumerate(traces))
    )
    print_section("CoT (code: JSON step loop until done)", body)
    return body


def demo_cot_sc(
    client: OpenAI,
    model: str,
    question: str,
    n_samples: int = 5,
    sample_temperature: float = 0.8,
) -> str:
    if n_samples < 1:
        print_section("CoT-SC", "Use --cot-sc-samples N with N >= 1.")
        return ""
    chains: list[str] = []
    votes: list[str] = []
    for i in range(n_samples):
        ans, traces = run_json_step_chain(
            client, model, question, temperature=sample_temperature
        )
        chains.append(
            f"--- run {i + 1}/{n_samples} (T={sample_temperature}) — final {ans!r} ---\n"
            + "\n---\n".join(traces)
        )
        votes.append(vote_normalize(ans) if ans != "(no conclusion)" else "(no conclusion)")

    counts = Counter(votes)
    winner, win_count = counts.most_common(1)[0]
    summary = (
        f"Code: ran {n_samples} independent JSON step chains; majority vote on parsed answers.\n"
        f"Votes: {dict(counts)}\nMajority: {winner!r} ({win_count}/{n_samples})\n\n"
        + "\n\n".join(chains)
    )
    print_section("CoT-SC (code: repeat JSON chain + vote)", summary)
    return summary


# --- ReAct: native tool calls; loop and dispatch in Python ---

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
            "description": "Evaluate a simple arithmetic expression (digits, + - * / . parentheses only).",
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
    log_prefix: str = "round",
) -> tuple[str | None, list[str], list[ChatCompletionMessageParam]]:
    """One ReAct-style episode: append tool results until `finish` or max_rounds."""
    messages: list[ChatCompletionMessageParam] = list(seed_messages)
    log_lines: list[str] = []
    final: str | None = None

    for round_i in range(max_rounds):
        resp = chat_completion(client, model, messages, temperature=0.1, tools=REACT_TOOLS)
        msg = resp.choices[0].message
        tcalls = msg.tool_calls or []

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if tcalls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tcalls
            ]
        messages.append(assistant_msg)  # type: ignore[arg-type]
        log_lines.append(
            f"--- {log_prefix} {round_i + 1} assistant ---\ncontent: {msg.content!r}\ntool_calls: {len(tcalls)}"
        )

        if not tcalls:
            break

        for tc in tcalls:
            out = dispatch_react_tool(tc.function.name, tc.function.arguments)
            if out.startswith("__FINISH__:"):
                final = out.removeprefix("__FINISH__:")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Recorded.",
                    }
                )
                log_lines.append(f"tool finish -> {final!r}")
                return final, log_lines, messages
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            log_lines.append(f"tool {tc.function.name} -> {out!r}")

    return final, log_lines, messages


def demo_react(client: OpenAI, model: str, question: str, max_rounds: int = 8) -> str:
    seed: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_MINIMAL},
        {"role": "user", "content": question},
    ]
    final, log_lines, _msgs = run_react_tool_loop(
        client, model, seed, max_rounds=max_rounds, log_prefix="round"
    )
    if final is not None:
        print_section(
            "ReAct (code: tool-call loop until finish)",
            "\n".join(log_lines) + f"\n\nDone: {final!r}",
        )
        return final
    print_section("ReAct (stopped — no finish tool)", "\n".join(log_lines))
    return "\n".join(log_lines)


# --- Plan-and-Act: JSON plan in code, then one tool episode per step ---


def demo_plan_act(
    client: OpenAI,
    model: str,
    question: str,
    *,
    max_rounds_per_step: int = 6,
) -> str:
    plan_payload = json.dumps({"task": question}, ensure_ascii=False) + (
        '\nReturn ONLY JSON: {"steps": [string]} — ordered substeps only, no tool use in this reply.'
    )
    plan_raw = chat_text(client, model, plan_payload, temperature=0.2)
    try:
        plan_obj = parse_json_object(plan_raw)
        raw_steps = plan_obj.get("steps") or []
    except (json.JSONDecodeError, ValueError):
        raw_steps = []

    steps = [str(s).strip() for s in raw_steps if s and str(s).strip()]
    if not steps:
        body = f"(no steps parsed)\nraw:\n{plan_raw}"
        print_section("Plan-and-Act", body)
        return body

    outcomes: list[str] = []
    blocks: list[str] = [f"=== PLAN ({len(steps)} steps) ===\n{json.dumps(steps, indent=2, ensure_ascii=False)}"]

    for i, step in enumerate(steps):
        act_payload = {
            "task": question,
            "plan_steps": steps,
            "step_index": i,
            "current_step": step,
            "prior_step_outputs": outcomes,
        }
        spec = (
            "Execute only current_step using the available functions. "
            "Call finish with a concise outcome for this step (intermediate value or note). "
            "If this is the last step (step_index == len(plan_steps)-1), finish with the final answer to task."
        )
        user_text = json.dumps(act_payload, ensure_ascii=False) + "\n" + spec
        seed = [
            {"role": "system", "content": SYSTEM_MINIMAL},
            {"role": "user", "content": user_text},
        ]
        step_final, step_logs, _ = run_react_tool_loop(
            client,
            model,
            seed,
            max_rounds=max_rounds_per_step,
            log_prefix=f"step {i + 1}/{len(steps)}",
        )
        blocks.append(
            f"=== ACT step {i + 1}/{len(steps)} ===\n{step}\n" + "\n".join(step_logs)
        )
        if step_final is None:
            outcomes.append(f"(no finish for step {i + 1})")
        else:
            outcomes.append(step_final)

    merge_payload = json.dumps(
        {"task": question, "step_outcomes": outcomes},
        ensure_ascii=False,
    ) + '\nReturn ONLY JSON: {"final_answer": string}.'
    merged_raw = chat_text(client, model, merge_payload, temperature=0.0)
    blocks.append("=== SYNTHESIZE (code always runs merge pass) ===\n" + merged_raw)

    try:
        final_ans = parse_json_object(merged_raw).get("final_answer", "")
    except (json.JSONDecodeError, ValueError):
        final_ans = outcomes[-1] if outcomes else ""

    full = "\n\n".join(blocks) + f"\n\n--- resolved final_answer ---\n{final_ans!r}"
    print_section("Plan-and-Act (plan JSON → per-step tool episodes → merge JSON)", full)
    return full


# --- PoT: tools python_run + finish; execution entirely in our process ---

POT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "python_run",
            "description": "Run a short Python snippet; stdout is returned. No network.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Full script body"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final answer after running any needed code.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


def run_pot_code(code: str, timeout: float = 3.0) -> str:
    body = "from __future__ import annotations\n" + code
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode != 0:
            return f"RuntimeError exit={proc.returncode}: {err or out or 'no output'}"
        return out or "(no stdout)"
    except subprocess.TimeoutExpired:
        return "Error: execution timeout."
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def dispatch_pot_tool(name: str, arguments: str) -> str:
    args = json.loads(arguments or "{}")
    if name == "python_run":
        return run_pot_code(str(args.get("code", "")))
    if name == "finish":
        return "__FINISH__:" + str(args.get("answer", ""))
    return "unknown tool"


def demo_pot(client: OpenAI, model: str, question: str, max_rounds: int = 6) -> str:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_MINIMAL},
        {"role": "user", "content": question},
    ]
    log_lines: list[str] = []
    final: str | None = None

    for round_i in range(max_rounds):
        resp = chat_completion(client, model, messages, temperature=0.1, tools=POT_TOOLS)
        msg = resp.choices[0].message
        tcalls = msg.tool_calls or []
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if tcalls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tcalls
            ]
        messages.append(assistant_msg)  # type: ignore[arg-type]
        log_lines.append(f"--- round {round_i + 1} ---\n{msg.content!r}\ntools: {len(tcalls)}")
        if not tcalls:
            break
        for tc in tcalls:
            out = dispatch_pot_tool(tc.function.name, tc.function.arguments)
            if out.startswith("__FINISH__:"):
                final = out.removeprefix("__FINISH__:")
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": "Recorded."}
                )
                log_lines.append(f"finish -> {final!r}")
                print_section(
                    "PoT (code: python_run tool + finish)",
                    "\n".join(log_lines),
                )
                return final or ""
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            log_lines.append(f"{tc.function.name} output: {out!r}")

    print_section("PoT (stopped without finish)", "\n".join(log_lines))
    return "\n".join(log_lines)


# --- ToT: branch / score / expand orchestrated in code ---


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


def _tot_expand_user(question: str, chosen: str) -> str:
    return (
        json.dumps({"task": question, "chosen_plan": chosen}, ensure_ascii=False)
        + "\n"
        + (
            "Solve using that plan. Reply with ONLY JSON: "
            '{"steps": [string], "answer": string}'
        )
    )


def demo_tot_shallow(client: OpenAI, model: str, question: str) -> str:
    """Degenerate ToT: one layer of branches + critic + single expansion (not a search tree)."""
    hypotheses: list[str] = []
    for k in range(3):
        raw = chat_text(
            client,
            model,
            _tot_hypothesis_user(question, k),
            temperature=0.45 + 0.1 * k,
        )
        try:
            hypotheses.append(str(parse_json_object(raw).get("hypothesis", "")))
        except (json.JSONDecodeError, ValueError):
            hypotheses.append(raw[:200])

    score_raw = chat_text(
        client,
        model,
        _tot_score_user(question, hypotheses),
        temperature=0.0,
    )
    best_index = 0
    try:
        sc = parse_json_object(score_raw)
        bi = sc.get("best_index", 0)
        if isinstance(bi, int) and 0 <= bi < len(hypotheses):
            best_index = bi
    except (json.JSONDecodeError, ValueError):
        pass

    chosen = hypotheses[best_index] if hypotheses else ""
    expand_raw = chat_text(
        client,
        model,
        _tot_expand_user(question, chosen),
        temperature=0.1,
    )
    body = textwrap.dedent(
        f"""\
        --- hypotheses ---
        {json.dumps(hypotheses, indent=2, ensure_ascii=False)}
        --- score raw ---
        {score_raw}
        --- chosen index ---
        {best_index}
        --- expand ---
        {expand_raw}
        """
    )
    print_section("ToT shallow (3 branches → score → expand; not tree search)", body)
    return body


def _tot_gen_thoughts(
    client: OpenAI,
    model: str,
    question: str,
    path: list[str],
    n: int,
    *,
    temperature: float,
) -> list[str]:
    payload = {"task": question, "prior_thoughts": path}
    spec = (
        f'Return ONLY JSON: {{"thoughts": [string]}} — exactly {n} distinct *next* '
        "inferences or sub-plans (short). If prior_thoughts is empty, propose starting directions."
    )
    raw = chat_text(
        client,
        model,
        json.dumps(payload, ensure_ascii=False) + "\n" + spec,
        temperature=temperature,
    )
    try:
        arr = parse_json_object(raw).get("thoughts") or []
        out = [str(x).strip() for x in arr if str(x).strip()]
        return out[:n] if len(out) >= n else out
    except (json.JSONDecodeError, ValueError):
        return []


def _tot_score_paths_batched(
    client: OpenAI,
    model: str,
    question: str,
    paths: list[list[str]],
) -> list[float]:
    if not paths:
        return []
    candidates = [{"thoughts": p} for p in paths]
    spec = (
        'Return ONLY JSON: {"scores": [number, ...]} — same length as candidates. '
        "Higher score = this partial reasoning is more likely to lead to a correct final answer."
    )
    user = json.dumps({"task": question, "candidates": candidates}, ensure_ascii=False) + "\n" + spec
    raw = chat_text(client, model, user, temperature=0.0)
    try:
        scores = parse_json_object(raw).get("scores") or []
        return [float(scores[i]) if i < len(scores) else 0.0 for i in range(len(paths))]
    except (json.JSONDecodeError, ValueError, TypeError, IndexError):
        return [0.0] * len(paths)


def _tot_finalize_path(client: OpenAI, model: str, question: str, path: list[str]) -> str:
    spec = 'Return ONLY JSON: {"answer": string}.'
    user = json.dumps({"task": question, "reasoning_path": path}, ensure_ascii=False) + "\n" + spec
    raw = chat_text(client, model, user, temperature=0.1)
    try:
        return str(parse_json_object(raw).get("answer", "")).strip()
    except (json.JSONDecodeError, ValueError):
        return raw[:500]


def demo_tot_bfs(
    client: OpenAI,
    model: str,
    question: str,
    *,
    branch_factor: int = 2,
    beam_width: int = 2,
    expand_rounds: int = 2,
) -> str:
    """
    Tree-of-Thoughts-style beam search: expand a frontier of partial traces,
    batch-score children, keep top beam (BFS layer-wise).
    """
    log: list[str] = []

    roots = _tot_gen_thoughts(client, model, question, [], branch_factor, temperature=0.5)
    if not roots:
        print_section("ToT BFS", "(no roots)\n")
        return ""
    frontier: list[list[str]] = [[t] for t in roots]
    scores = _tot_score_paths_batched(client, model, question, frontier)
    ranked = sorted(zip(scores, frontier), key=lambda x: -x[0])
    frontier = [p for _, p in ranked[:beam_width]]
    log.append(
        "--- layer 0 ---\n"
        + json.dumps(
            [{"path": p, "score": s} for s, p in ranked[:beam_width]],
            indent=2,
            ensure_ascii=False,
        )
    )

    for layer in range(expand_rounds):
        children: list[list[str]] = []
        for path in frontier:
            cont = _tot_gen_thoughts(
                client, model, question, path, branch_factor, temperature=0.35
            )
            for c in cont:
                children.append(path + [c])
        if not children:
            log.append(f"--- layer {layer + 1}: no children ---")
            break
        sc = _tot_score_paths_batched(client, model, question, children)
        ranked = sorted(zip(sc, children), key=lambda x: -x[0])
        frontier = [p for _, p in ranked[:beam_width]]
        log.append(
            f"--- layer {layer + 1} (expand + beam) ---\n"
            + json.dumps(
                [{"path": p, "score": s} for s, p in ranked[:beam_width]],
                indent=2,
                ensure_ascii=False,
            )
        )

    best_path = frontier[0]
    sc_front = _tot_score_paths_batched(client, model, question, frontier)
    if len(frontier) > 1 and sc_front:
        pick = max(range(len(frontier)), key=lambda i: sc_front[i] if i < len(sc_front) else 0.0)
        best_path = frontier[pick]
    final_ans = _tot_finalize_path(client, model, question, best_path)
    log.append("--- best path ---\n" + json.dumps(best_path, indent=2, ensure_ascii=False))
    log.append(f"--- final ---\n{final_ans!r}")

    body = "\n\n".join(log)
    print_section("ToT beam-BFS (tree frontier + batched value scores + prune)", body)
    return body


def demo_tot_dfs(
    client: OpenAI,
    model: str,
    question: str,
    *,
    branch_factor: int = 2,
    max_depth: int = 4,
    prune_below: float = 3.0,
) -> str:
    """Depth-first: expand highest-scored child first; prune branches below prune_below."""
    log: list[str] = []
    best_path: list[str] = []
    best_score = float("-inf")

    def dfs(path: list[str], depth: int) -> None:
        nonlocal best_path, best_score
        if depth >= max_depth:
            s = _tot_score_paths_batched(client, model, question, [path])
            sc = s[0] if s else 0.0
            log.append(f"leaf depth={depth} score={sc:.2f} path={path!r}")
            if sc > best_score:
                best_score, best_path = sc, list(path)
            return
        kids = _tot_gen_thoughts(client, model, question, path, branch_factor, temperature=0.4)
        if not kids:
            return
        child_paths = [path + [k] for k in kids]
        scores = _tot_score_paths_batched(client, model, question, child_paths)
        order = sorted(range(len(child_paths)), key=lambda i: -scores[i])
        for idx in order:
            if scores[idx] < prune_below:
                log.append(
                    f"prune depth={depth} score={scores[idx]:.2f} < {prune_below} (stop siblings worse)"
                )
                break
            dfs(child_paths[idx], depth + 1)

    dfs([], 0)
    if not best_path:
        fb = _tot_gen_thoughts(client, model, question, [], 1, temperature=0.3)
        best_path = [fb[0]] if fb else []
    final_ans = _tot_finalize_path(client, model, question, best_path) if best_path else ""
    body = (
        "\n".join(log)
        + f"\n\n--- best_path (score={best_score}) ---\n{best_path!r}\n\nfinal: {final_ans!r}"
    )
    print_section("ToT DFS + prune (depth-first, low-value branches dropped)", body)
    return body


def demo_tot(client: OpenAI, model: str, question: str) -> str:
    """Default ToT: beam-BFS tree search (see tot_shallow / tot_dfs)."""
    return demo_tot_bfs(client, model, question)


# --- MoR: Memory over Reasoning — external memory + retrieval window; compact in code ---


def _mor_fold_facts(client: OpenAI, model: str, stale_facts: list[str], prev_summary: str) -> str:
    """Code-triggered compaction: fold older facts into a short summary line."""
    payload = {
        "previous_summary": prev_summary,
        "facts_to_fold": stale_facts,
    }
    spec = (
        "Compress the facts into one short bullet for long-term summary. "
        'Return ONLY JSON: {"compressed": string}.'
    )
    raw = chat_text(
        client,
        model,
        json.dumps(payload, ensure_ascii=False) + "\n" + spec,
        temperature=0.0,
    )
    try:
        return str(parse_json_object(raw).get("compressed", "")).strip()
    except (json.JSONDecodeError, ValueError):
        return ""


def demo_mor(
    client: OpenAI,
    model: str,
    question: str,
    *,
    max_steps: int = 10,
    fact_window: int = 6,
    compact_after: int = 10,
    fold_batch: int = 4,
) -> str:
    """
    MoR (Memory over Reasoning): each turn the model sees **task + memory slice** only,
    not the full raw reasoning log. Python owns the store, retrieval window, and when to compact.
    """
    memory: dict[str, Any] = {"summary": "", "facts": []}

    log: list[str] = []
    final: str | None = None

    for turn in range(max_steps):
        facts: list[str] = memory["facts"]
        if len(facts) > compact_after:
            stale = facts[:fold_batch]
            memory["facts"] = facts[fold_batch:]
            folded = _mor_fold_facts(client, model, stale, memory["summary"])
            if folded:
                memory["summary"] = (memory["summary"] + " " + folded).strip()[-1500:]

        recent = memory["facts"][-fact_window:]
        payload = {
            "task": question,
            "memory_visible": {
                "summary": memory["summary"],
                "recent_facts": recent,
            },
        }
        spec = (
            "Keep reasoning_brief short; put durable information in add_facts. "
            "Return ONLY JSON with keys: "
            '"reasoning_brief" (string, optional short note), '
            '"add_facts" (array of strings to remember), '
            '"done" (boolean), '
            '"answer" (string or null when not done).'
        )
        raw = chat_text(
            client,
            model,
            json.dumps(payload, ensure_ascii=False) + "\n" + spec,
            temperature=0.2,
        )
        log.append(f"--- turn {turn + 1} ---\n{raw}")
        try:
            data = parse_json_object(raw)
        except (json.JSONDecodeError, ValueError):
            continue

        for f in data.get("add_facts") or []:
            s = str(f).strip()
            if s:
                memory["facts"].append(s)

        if bool(data.get("done")):
            ans = data.get("answer")
            if ans is not None and str(ans).strip():
                final = str(ans).strip()
                break

    mem_dump = json.dumps(memory, indent=2, ensure_ascii=False)
    footer = f"\n\n--- memory store at end ---\n{mem_dump}\n\n--- final ---\n{final!r}"
    body = "\n".join(log) + footer
    print_section("MoR (Memory over Reasoning: store + window + compaction in code)", body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="Reasoning pattern demos (orchestration in code)")
    parser.add_argument(
        "--demo",
        choices=[
            "cot",
            "cot_sc",
            "react",
            "plan_act",
            "pot",
            "tot",
            "tot_shallow",
            "tot_dfs",
            "mor",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Task for all demos")
    parser.add_argument("--cot-sc-samples", type=int, default=5, help="CoT-SC chain count")
    parser.add_argument(
        "--cot-sc-temperature",
        type=float,
        default=0.8,
        help="Temperature for each CoT-SC chain",
    )
    args = parser.parse_args()

    client = build_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    demos: dict[str, Callable[[OpenAI, str, str], str]] = {
        "cot": demo_cot,
        "react": demo_react,
        "plan_act": demo_plan_act,
        "pot": demo_pot,
        "tot": demo_tot,
        "tot_shallow": demo_tot_shallow,
        "tot_dfs": demo_tot_dfs,
        "mor": demo_mor,
    }

    order = ["cot", "cot_sc", "react", "plan_act", "pot", "tot", "mor"]

    def run_one(name: str) -> str:
        if name == "cot_sc":
            return demo_cot_sc(
                client,
                model,
                args.question,
                n_samples=args.cot_sc_samples,
                sample_temperature=args.cot_sc_temperature,
            )
        return demos[name](client, model, args.question)

    if args.demo == "all":
        for name in order:
            print(f"\n>>> Running {name.upper()}...")
            run_one(name)
    else:
        run_one(args.demo)


if __name__ == "__main__":
    main()
