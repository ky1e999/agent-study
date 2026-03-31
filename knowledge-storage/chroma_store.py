"""ChromaDB: one persistent collection per embedding lane (dimension must match model)."""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, EmbeddingLane


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection(client, lane: EmbeddingLane) -> Collection:
    return client.get_or_create_collection(
        name=lane.chroma_collection,
        metadata={
            "embedding_model": lane.model_name,
            "lane_id": lane.lane_id,
            "embedding_dimension": str(lane.embedding_dimension),
        },
    )


def upsert_lane(
    lane: EmbeddingLane,
    chunk_ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    if not chunk_ids:
        return
    model = SentenceTransformer(lane.model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
    ).tolist()
    client = get_client()
    collection = get_collection(client, lane)
    collection.upsert(ids=chunk_ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def query_lane(
    lane: EmbeddingLane,
    query_text: str,
    n_results: int = 5,
) -> dict[str, Any]:
    model = SentenceTransformer(lane.model_name)
    q_emb = model.encode([query_text], convert_to_numpy=True).tolist()
    client = get_client()
    collection = get_collection(client, lane)
    return collection.query(query_embeddings=q_emb, n_results=n_results)
