"""SQLite metadata: documents and chunks (text lives here; vectors live in Chroma)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import EmbeddingLane


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
    UNIQUE (doc_id, chunk_index)
);

-- Registry: which embedding model / Chroma collection each lane uses (query must match).
CREATE TABLE IF NOT EXISTS embedding_lanes (
    lane_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    chroma_collection TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL
);

-- Per-document index audit: this doc was embedded with this lane into that collection.
CREATE TABLE IF NOT EXISTS document_lane_index (
    doc_id TEXT NOT NULL,
    lane_id TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    PRIMARY KEY (doc_id, lane_id),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
    FOREIGN KEY (lane_id) REFERENCES embedding_lanes(lane_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_lane_doc ON document_lane_index(doc_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str


def insert_document(conn: sqlite3.Connection, doc_id: str, source_path: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO documents (doc_id, source_path, ingested_at) VALUES (?, ?, ?)",
        (doc_id, source_path, now),
    )


def insert_chunks(conn: sqlite3.Connection, chunks: list[ChunkRecord]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO chunks (chunk_id, doc_id, chunk_index, text)
        VALUES (?, ?, ?, ?)
        """,
        [(c.chunk_id, c.doc_id, c.chunk_index, c.text) for c in chunks],
    )


def upsert_embedding_lane(
    conn: sqlite3.Connection,
    lane_id: str,
    model_name: str,
    chroma_collection: str,
    embedding_dimension: int,
) -> None:
    conn.execute(
        """
        INSERT INTO embedding_lanes (lane_id, model_name, chroma_collection, embedding_dimension)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(lane_id) DO UPDATE SET
            model_name = excluded.model_name,
            chroma_collection = excluded.chroma_collection,
            embedding_dimension = excluded.embedding_dimension
        """,
        (lane_id, model_name, chroma_collection, embedding_dimension),
    )


def record_document_lane_index(
    conn: sqlite3.Connection,
    doc_id: str,
    lane_id: str,
    chunk_count: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO document_lane_index (doc_id, lane_id, indexed_at, chunk_count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(doc_id, lane_id) DO UPDATE SET
            indexed_at = excluded.indexed_at,
            chunk_count = excluded.chunk_count
        """,
        (doc_id, lane_id, now, chunk_count),
    )


def fetch_all_chunks(conn: sqlite3.Connection, doc_id: str | None = None) -> list[ChunkRecord]:
    if doc_id:
        cur = conn.execute(
            "SELECT chunk_id, doc_id, chunk_index, text FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        )
    else:
        cur = conn.execute(
            "SELECT chunk_id, doc_id, chunk_index, text FROM chunks ORDER BY doc_id, chunk_index"
        )
    return [ChunkRecord(**dict(row)) for row in cur.fetchall()]


def get_chunk_text(conn: sqlite3.Connection, chunk_id: str) -> str | None:
    row = conn.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    return row["text"] if row else None


def resolve_lane_for_doc_model(
    conn: sqlite3.Connection,
    doc_id: str,
    model_name: str,
) -> EmbeddingLane | None:
    """Look up the retrieval lane for a document and embedding model (must exist in document_lane_index)."""
    row = conn.execute(
        """
        SELECT el.lane_id, el.model_name, el.chroma_collection, el.embedding_dimension
        FROM embedding_lanes el
        INNER JOIN document_lane_index dli ON el.lane_id = dli.lane_id
        WHERE dli.doc_id = ? AND el.model_name = ?
        """,
        (doc_id, model_name),
    ).fetchone()
    if row is None:
        return None
    return EmbeddingLane(
        lane_id=row["lane_id"],
        model_name=row["model_name"],
        chroma_collection=row["chroma_collection"],
        embedding_dimension=int(row["embedding_dimension"]),
    )
