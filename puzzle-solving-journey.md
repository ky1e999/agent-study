# Puzzle-solving journey (draft)

**Purpose:** Track questions about agents, RAG, embeddings, and retrieval—before filling in answers.

**Status key**

- [ ] Not started
- [~] In progress
- [x] Answered (add notes link or section below the puzzle)

**Last updated:** 2026-03-29

---

## 1. Implement CoT, ReAct, ToT in code

- [x] Draft answer
- [x] Code sketch / references (libs, patterns)

**Notes:**

Working implementations live in [`reasoning-demos/`](reasoning-demos/README.md) (`run_demos.py`). Patterns are **orchestrated in Python** (loops, tool dispatch, beam BFS / DFS, voting), not by “mode switching” prose in the system prompt—see README table for CoT, CoT-SC, ReAct, Plan-and-Act, PoT, ToT (`tot` = beam-BFS; `tot_shallow`, `tot_dfs`), MoR (memory-over-reasoning).

---

## 2. Combining Chain-of-Thought / Tree-of-Thought with ReAct

- [x] Draft answer
- [x] Architecture sketch (when to nest, when to alternate)

**Notes:**

Runnable combos in [`reasoning-combination/`](reasoning-combination/README.md) (`run_combos.py`): **CoT→ReAct**, **interleaved** CoT/ReAct, **ToT→ReAct**, **ToT+CoT→ReAct**, **ToT→ReAct→replan**, **GoT (DAG)→ReAct** per node + merge. Same idea throughout: **ReAct** = tool loop; **CoT/ToT/GoT** = structured deliberation **before** or **between** bursts—see README mode table and optional LLM trace for prompts.

---

## 3. Word embedding vs sentence embedding


- [ ] Draft answer
- [ ] When to use which

**Notes (fill later):**

---

## 4. Switching embedding model A → B after indexing

- [ ] Draft answer
- [ ] Migration / dual-write / re-embed strategy

**Notes (fill later):**

---

## 5. How to choose top-k for RAG and rerank

### 5.1 Retrieval K

- [ ] Hit rate
- [ ] Recall rate
- [ ] Latency

### 5.2 Rerank K

- [ ] Precision
- [ ] Token cost
- [ ] MRR
- [ ] nDCG

**Notes (fill later):**

---

## 6. Assessing reranker validity and metrics

- [ ] Draft answer
- [ ] Metrics list + eval setup

**Notes (fill later):**

---

## 7. Chunk size and overlap

### 7.1 Retrieval recall

- [ ] Is the “right” answer actually in retrieved chunks?

### 7.2 Faithfulness

- [ ] Is generation grounded in chunks vs hallucination?

- [ ] **TODO:** Benchmark on project training dataset

**Notes (fill later):**

---

## 8. Updating K dynamically

- [ ] Draft answer
- [ ] Heuristics / adaptive strategies

**Notes (fill later):**

---

## 9. Why we can’t extract K and V during RAG

_(Clarify later: KV cache in transformers vs “keys/values” in attention vs something else.)_

- [ ] Draft answer

**Notes (fill later):**

---

## 10. Secure token amount vs compact token amount

- [ ] Draft answer
- [ ] Definitions in our stack (if product-specific)

**Notes (fill later):**

---

## 11. Vector DB compact after many user inserts — choosing N

- [ ] Draft answer
- [ ] Signals (memory, fragmentation, merge policies)

**Notes (fill later):**

---

## 12. When to do episode digesting

- [ ] Draft answer
- [ ] Triggers (length, time, importance)

**Notes (fill later):**

---

## 13. What is a knowledge graph?

- [ ] Draft answer
- [ ] Relation to RAG / agents

**Notes (fill later):**

---

## Session log

| Date       | Puzzle IDs touched | Summary |
| ---------- | ------------------ | ------- |
| 2026-03-29 | —                  | Created draft tracker |
| 2026-03-29 | 1                  | Reasoning demos (CoT, ReAct, ToT, etc.); puzzle 1 marked answered |
| 2026-03-29 | 2                  | reasoning-combination (CoT/ToT/GoT + ReAct); puzzle 2 marked answered |
