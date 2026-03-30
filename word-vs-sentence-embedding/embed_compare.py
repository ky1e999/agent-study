"""Compare sentence-level encoding vs naive word-bag pooling using the same backbone model."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1-D vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def tokenize_words(text: str) -> list[str]:
    """Simple whitespace + punctuation split; keeps alphanumerics per token."""
    raw = re.findall(r"[A-Za-z0-9']+", text.lower())
    return [t for t in raw if t]


class EmbeddingComparer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode_sentence_level(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return _normalize_rows(np.asarray(emb, dtype=np.float64))

    def encode_word_bag(self, texts: Sequence[str]) -> np.ndarray:
        """
        Mean of encodings of each *word* as its own mini-sentence.
        Loses order, collocation, and most multi-word meaning — useful as a baseline.
        """
        out: list[np.ndarray] = []
        for text in texts:
            words = tokenize_words(text)
            if not words:
                out.append(np.zeros(self.model.get_sentence_embedding_dimension()))
                continue
            w_emb = self.model.encode(
                words,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            out.append(np.mean(np.asarray(w_emb, dtype=np.float64), axis=0))
        return _normalize_rows(np.stack(out, axis=0))

    def pair_similarities(
        self,
        texts_a: Sequence[str],
        texts_b: Sequence[str],
        *,
        mode: str,
    ) -> np.ndarray:
        if mode == "sentence":
            ea = self.encode_sentence_level(texts_a)
            eb = self.encode_sentence_level(texts_b)
        elif mode == "word_bag":
            ea = self.encode_word_bag(texts_a)
            eb = self.encode_word_bag(texts_b)
        else:
            raise ValueError("mode must be 'sentence' or 'word_bag'")
        return np.sum(ea * eb, axis=1)
