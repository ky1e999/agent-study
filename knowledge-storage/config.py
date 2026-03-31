"""Paths and embedding lanes for the knowledge-storage demo."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

# agent-study/knowledge-storage -> agent-workspace
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_PDF = _REPO_ROOT / "rag-all-techniques" / "data" / "AI_Information.en.zh-CN.pdf"

# Local persistence (relative to this package directory)
PACKAGE_DIR = Path(__file__).resolve().parent
SQLITE_PATH = PACKAGE_DIR / "knowledge_metadata.db"
CHROMA_PATH = PACKAGE_DIR / "chroma_data"

# Placeholders — tune via benchmarks later
CHUNK_CHAR_SIZE = 800
CHUNK_CHAR_OVERLAP = 100


@dataclass(frozen=True)
class EmbeddingLane:
    """One retrieval lane: model + Chroma collection (fixed dimension per collection)."""

    lane_id: str
    model_name: str
    chroma_collection: str
    embedding_dimension: int


# Different output dimensions: 384 vs 768
LANES: tuple[EmbeddingLane, ...] = (
    EmbeddingLane(
        lane_id="minilm-l6-384",
        model_name="all-MiniLM-L6-v2",
        chroma_collection="kb_minilm_l6_v2",
        embedding_dimension=384,
    ),
    EmbeddingLane(
        lane_id="mpnet-base-768",
        model_name="all-mpnet-base-v2",
        chroma_collection="kb_mpnet_base_v2",
        embedding_dimension=768,
    ),
)

LANE_BY_ID: dict[str, EmbeddingLane] = {lane.lane_id: lane for lane in LANES}
