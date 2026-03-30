#!/usr/bin/env python3
"""
Benchmark sentence-level vs word-bag embeddings on a JSON dataset.

Metrics:
  - mean cosine on paraphrase pairs (higher is better)
  - mean cosine on unrelated pairs (lower is better)
  - separation = paraphrase_mean - unrelated_mean (higher is better)
  - context_clash mean cosine (polysemy: ideally not as high as paraphrases)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from embed_compare import EmbeddingComparer


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def collect_pairs(section: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    a_list: list[str] = []
    b_list: list[str] = []
    for row in section:
        a_list.append(row["a"])
        b_list.append(row["b"])
    return a_list, b_list


def summarize(
    name: str,
    sims: Any,
) -> None:
    arr = np.asarray(sims, dtype=float)
    print(f"  {name}: mean={arr.mean():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sentence vs word-bag embeddings")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "benchmark_pairs.json",
        help="Path to benchmark_pairs.json",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model id",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"Data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    blob = load_json(args.data)
    para_a, para_b = collect_pairs(blob["paraphrases"])
    un_a, un_b = collect_pairs(blob["unrelated"])
    clash_a, clash_b = collect_pairs(blob["context_clash"])

    print(f"Model: {args.model}")
    print("Loading …")
    cmp = EmbeddingComparer(args.model)

    for mode in ("sentence", "word_bag"):
        print(f"\n=== mode={mode} ===")
        p_sim = cmp.pair_similarities(para_a, para_b, mode=mode)
        u_sim = cmp.pair_similarities(un_a, un_b, mode=mode)
        c_sim = cmp.pair_similarities(clash_a, clash_b, mode=mode)
        summarize("paraphrases", p_sim)
        summarize("unrelated", u_sim)
        summarize("context_clash", c_sim)
        sep = float(p_sim.mean() - u_sim.mean())
        print(f"  separation (para_mean - unrelated_mean): {sep:.4f}")

    print(
        "\nInterpretation: Strong sentence encoding usually **raises separation** and "
        "keeps **context_clash** similarities lower than paraphrase similarities; "
        "word-bag baselines often blur polysemy and miss phrasing."
    )


if __name__ == "__main__":
    main()
