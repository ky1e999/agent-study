#!/usr/bin/env python3
"""Small interactive demo: sentence vs word-bag similarity on a few pairs."""

from __future__ import annotations

import argparse

from embed_compare import EmbeddingComparer


def main() -> None:
    parser = argparse.ArgumentParser(description="Word-bag vs sentence embedding demo")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model id (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    pairs = [
        (
            "We sat on the bank watching the ducks float by.",
            "The investment bank approved the acquisition loan.",
        ),
        (
            "Machine learning models need clean training data.",
            "To train ML models you need well-prepared data.",
        ),
        (
            "How do I renew my passport?",
            "Gradient descent minimizes the loss function.",
        ),
    ]

    print(f"Loading model {args.model} …")
    cmp = EmbeddingComparer(args.model)

    print("\nPair | sentence cos | word-bag cos")
    print("-" * 52)
    for a, b in pairs:
        s_sim = float(cmp.pair_similarities([a], [b], mode="sentence")[0])
        w_sim = float(cmp.pair_similarities([a], [b], mode="word_bag")[0])
        short = (a[:36] + "…") if len(a) > 36 else a
        print(f"{short:38} | {s_sim:10.4f} | {w_sim:10.4f}")

    print(
        "\nExpectation: paraphrases score higher with **sentence** encoding; "
        "polysemous pairs stay more separated than word-bag often allows."
    )


if __name__ == "__main__":
    main()
