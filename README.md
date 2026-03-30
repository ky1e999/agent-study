# Agent study

Personal notes and small runnable demos around agents, RAG, embeddings, and retrieval. The question list and status live in [`puzzle-solving-journey.md`](puzzle-solving-journey.md).

## Dense RAG: index and query must share the same embedding stack

For typical **dense vector retrieval** (one embedding per chunk, same similarity as in training—e.g. cosine or dot product after a fixed normalization), **knowledge-base chunks and user queries must be embedded with the same configuration**. If the index was built in embedding space A and you embed the query in space B, you are comparing vectors that usually do not live in a comparable geometry, so **ranking and similarity scores are not meaningful** for a naïve mix-and-match.

**Necessity — keep these aligned for every index “lane”:**

- Same **model id/version**
- Same **prompts/instructions** if the API supports them (e.g. asymmetric prefixes for “query:” vs “passage:”)
- Same **normalization** (e.g. L2 to unit length)
- Same **max length / truncation** behavior

Treat **(model, version, preprocessing)** as a single versioned unit. Changing any of these on only one side—without re-embedding the whole corpus in the new space—is a frequent cause of **silent retrieval regressions**. Migrating to a new model is covered conceptually under puzzle **§4** in [`puzzle-solving-journey.md`](puzzle-solving-journey.md).

## Subprojects

| Folder | Topic |
|--------|--------|
| [`reasoning-demos/`](reasoning-demos/README.md) | CoT, ReAct, ToT-style patterns |
| [`reasoning-combination/`](reasoning-combination/README.md) | Nesting / interleaving reasoning + tools |
| [`word-vs-sentence-embedding/`](word-vs-sentence-embedding/README.md) | Sentence vs word-bag pooling (puzzle #3) |
