# Reasoning pattern demos (CoT, CoT-SC, ReAct, Plan-and-Act, PoT, ToT, MoR)

Patterns are **orchestrated in Python** (loops, tool dispatch, voting, multi-run merge). Prompts stay minimal: a generic assistant system line plus **structured user payloads** or **native function calling**—not “switch into ReAct / CoT mode” prose.

| Pattern | What the code does |
|--------|---------------------|
| **CoT** | **JSON step loop:** each turn sends `{task, prior_steps}`; model returns `{step, done, answer}`; repeat until `done`. |
| **CoT-SC** | Run that full loop **N times** (higher temperature) and **majority-vote** parsed final answers. |
| **ReAct** | **Tool-calling API loop:** `search` / `calc` / `finish`; your code executes tools and appends results until `finish`. |
| **Plan-and-Act** | **Phase 1:** model returns JSON `{"steps":[...]}` only. **Phase 2:** for each step, a fresh **tool episode** (same tools as ReAct); **Phase 3:** code runs a short JSON merge for `final_answer`. |
| **PoT** | **Tool loop:** `python_run(code)` (subprocess) + `finish`; execution stays in your process. |
| **ToT** (`tot`) | **Beam-BFS on a reasoning tree:** expand each frontier path into `branch_factor` children → **batch-score** all children → keep top **beam** (layer-wise). Then finalize best path. |
| **ToT shallow** (`tot_shallow`) | Older **flat** demo: 3 hypotheses → one score → one expand (not real tree search). |
| **ToT DFS** (`tot_dfs`) | **Depth-first:** expand children sorted by score; **prune** below a threshold (skip weak branches). |
| **MoR** | **Memory over reasoning:** Python keeps a **memory store** (summary + facts); each turn only a **window** is shown; **compaction** folds old facts into summary—avoid shipping the full scratchpad. |

## Requirements

- **ReAct**, **Plan-and-Act** (act phase), and **PoT** need a model + endpoint that supports **OpenAI-style `tool_calls`** (function calling).
- **CoT / ToT / MoR** only need chat completions; outputs must be **parseable JSON** (the script strips ``` fences if present).

## Security

- Put your API key in `.env`. Never commit `.env`.
- If a key was pasted into a chat, treat it as leaked and **rotate** it.

## Setup

```bash
cd reasoning-demos
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp env.example .env
# Edit .env with OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
```

## Run

```bash
python run_demos.py --demo all
python run_demos.py --demo cot
python run_demos.py --demo cot_sc
python run_demos.py --demo react
python run_demos.py --demo plan_act
python run_demos.py --demo pot
python run_demos.py --demo tot
python run_demos.py --demo tot_shallow
python run_demos.py --demo tot_dfs
python run_demos.py --demo mor
```

Optional: `--question ...` (defaults to a small math/word problem). CoT-SC: `--cot-sc-samples` (default 5), `--cot-sc-temperature` (default 0.8).

`python_run` uses a **subprocess with a short timeout** (demo only; tighten for production).
