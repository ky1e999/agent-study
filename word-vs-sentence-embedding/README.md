# Word vs sentence embeddings (demo + benchmark)

Small study project for **puzzle #3** in [`puzzle-solving-journey.md`](../puzzle-solving-journey.md): same `SentenceTransformer` backbone, two pooling strategies, and a JSON benchmark to see which behaves more sensibly.

## Two modes (same model)

| Mode | What it does |
|------|----------------|
| **Sentence** | Encode the **full string** once → one vector per text. Context, word order, and multi-word meaning stay in the representation. |
| **Word-bag** | Split into **words**, encode **each word alone** (no neighboring context), **average** those vectors → one vector per text. Order and phrasing are mostly thrown away; shared surface words (e.g. “bank”) dominate the average. |

After encoding, vectors are L2-normalized; **cosine similarity** between a pair is the dot product (implemented in `embed_compare.py`).

## How to read the benchmark (important)

**Higher cosine is not automatically “better.”** You care about **discrimination**:

- **Paraphrases** should score **high** (same meaning).
- **Unrelated** pairs should score **low** (different topics).
- **Separation** = mean(paraphrases) − mean(unrelated). **Larger is better** for telling “same meaning” from “random.”

Typical pattern (e.g. `all-MiniLM-L6-v2` on `data/benchmark_pairs.json`):

- **Sentence** mode: unrelated cosine stays **small**, separation **large**.
- **Word-bag** mode: many pairs look **similar** (unrelated cosine stays **high**); **context_clash** (polysemy) can even look **as high as paraphrases**—that is **worse**, not better.

So word-bag often **inflates** similarity across the board; it is a **pedagogical baseline**, not proof that “word embeddings always win.”

## When to use which (practically)

**Prefer sentence (or chunk / document) embeddings when:**

- Semantic search, RAG, dedup, clustering **passages** or **tickets**.
- Paraphrases, user questions vs docs—**syntax and context** matter.

**Word-level embeddings (classic word2vec/GloVe, or token-centric vectors) when:**

- The unit of meaning is really a **word** or **label**: lexical similarity, vocabulary clustering, some feature pipelines.
- You explicitly need **fixed per-type vectors** or very lightweight / legacy stacks.

This repo’s **word-bag** path is: sentence model + one-word-at-a-time + mean. It illustrates **smearing** and polysemy; it is not the same as training dedicated word vectors.

## Setup

From the `agent-study` root (shared `requirements.txt`):

```bash
cd /path/to/agent-study
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First model download may need network access.

## Run

```bash
# From agent-study/
python word-vs-sentence-embedding/demo.py
python word-vs-sentence-embedding/benchmark_embed.py
# Optional: --model another sentence-transformers id
```

## Files

| File | Role |
|------|------|
| `embed_compare.py` | `EmbeddingComparer`: `encode_sentence_level`, `encode_word_bag`, `pair_similarities`. |
| `demo.py` | Few hand-picked pairs; prints sentence vs word-bag cosine side by side. |
| `benchmark_embed.py` | Loads `data/benchmark_pairs.json`; reports means, separation, both modes. |
| `data/benchmark_pairs.json` | `paraphrases`, `unrelated`, `context_clash` pairs. |
