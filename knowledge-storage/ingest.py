"""Ingest a PDF: SQLite chunks + Chroma upserts for each embedding lane."""

from __future__ import annotations

from pathlib import Path

from config import DEFAULT_PDF, LANES, SQLITE_PATH, EmbeddingLane
from chroma_store import upsert_lane
from pdf_chunking import chunk_text, extract_pdf_text
from sqlite_store import (
    ChunkRecord,
    connect,
    init_db,
    insert_chunks,
    insert_document,
    record_document_lane_index,
    upsert_embedding_lane,
)


def doc_id_from_path(path: Path) -> str:
    stem = path.stem.replace(".", "_")
    return f"doc_{stem}"


def ingest_pdf(
    pdf_path: Path,
    sqlite_path: Path = SQLITE_PATH,
    lanes: tuple[EmbeddingLane, ...] = LANES,
    clear_sqlite_chunks_for_doc: bool = True,
) -> str:
    pdf_path = pdf_path.resolve()
    doc_id = doc_id_from_path(pdf_path)
    raw = extract_pdf_text(pdf_path)
    pieces = chunk_text(raw)
    if not pieces:
        raise RuntimeError(f"No text extracted from {pdf_path}")

    init_db(sqlite_path)
    records = [
        ChunkRecord(
            chunk_id=f"{doc_id}#{idx:05d}",
            doc_id=doc_id,
            chunk_index=idx,
            text=text,
        )
        for idx, text in enumerate(pieces)
    ]

    with connect(sqlite_path) as conn:
        for lane in lanes:
            upsert_embedding_lane(
                conn,
                lane.lane_id,
                lane.model_name,
                lane.chroma_collection,
                lane.embedding_dimension,
            )
        if clear_sqlite_chunks_for_doc:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        insert_document(conn, doc_id, str(pdf_path))
        insert_chunks(conn, records)
        conn.commit()

    chunk_ids = [r.chunk_id for r in records]
    texts = [r.text for r in records]
    base_meta = [{"doc_id": doc_id, "chunk_index": r.chunk_index} for r in records]

    for lane in lanes:
        metadatas = [
            {
                **m,
                "embedding_lane": lane.lane_id,
                "embedding_model": lane.model_name,
            }
            for m in base_meta
        ]
        upsert_lane(lane, chunk_ids, texts, metadatas)
        with connect(sqlite_path) as conn:
            record_document_lane_index(conn, doc_id, lane.lane_id, len(records))
            conn.commit()

    return doc_id


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Ingest PDF into SQLite + Chroma lanes.")
    p.add_argument(
        "pdf",
        nargs="?",
        default=str(DEFAULT_PDF),
        help="Path to PDF (default: AI_Information.en.zh-CN.pdf in rag-all-techniques)",
    )
    args = p.parse_args()
    did = ingest_pdf(Path(args.pdf))
    print(f"ingested doc_id={did}, chunks written to {SQLITE_PATH} and Chroma under {Path(__file__).resolve().parent / 'chroma_data'}")
