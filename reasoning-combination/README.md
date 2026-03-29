# Reasoning combinations (ReAct × CoT / ToT / GoT)

Runs **baseline ReAct** and combined pipelines so you can compare answers, traces, and wall time.

## Setup

From `agent-study/` (where `.env` may live) or this folder:

```bash
cd reasoning-combination
pip install -r requirements.txt
```

Ensure `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` are set (same as `reasoning-demos`).

## Run

```bash
python run_combos.py --combo all
python run_combos.py --combo cot_then_react
python run_combos.py --combo tot_react_replan --question "Your task..."
```

## Modes

| `--combo` | Idea |
|-----------|------|
| `react_only` | Tool loop only (baseline). |
| `cot_then_react` | JSON reasoning-step list (no tools), then ReAct with that context. |
| `cot_interleaved_react` | Cycles: one CoT-style JSON step → short ReAct burst → feed tool trace back. |
| `tot_then_react` | 3 hypotheses + critic → chosen approach → ReAct. |
| `tot_cot_then_react` | ToT pick → JSON **CoT steps** under that strategy → ReAct. |
| `tot_react_replan` | ToT → ReAct; if no `finish`, second ToT with failure context → ReAct again. |
| `got_react` | LLM emits a small DAG of subgoals → topo order → ReAct per node → merge answer. |

Timing prints as `[combo] 12.34s final=...`.

### LLM trace (prompt + context each call)

By default every API call prints:

- A running **`LLM call #N`** with a **label** (e.g. `ReAct/c0 round 2`, `ToT hypothesis k=0`).
- **Full context**: `system` + `user`, or for ReAct the **entire message list** so far (including prior assistant turns and **tool results**).
- A short **assistant reply** (or tool-call summary) after the response.

For `--combo all`, output is huge; use **`--quiet-prompts`** on that sweep:

```bash
python run_combos.py --combo all --quiet-prompts
python run_combos.py --combo tot_then_react
```
