"""
Demo: ingest the AI information PDF, then query each embedding lane.

Queries must use the same model / collection as the index (see agent-study/README.md).
"""

from __future__ import annotations

from pathlib import Path

from config import DEFAULT_PDF, EmbeddingLane, SQLITE_PATH
from chroma_store import query_lane
from ingest import ingest_pdf
from sqlite_store import connect, fetch_all_chunks, init_db, resolve_lane_for_doc_model


def require_lane(conn, doc_id: str, model_name: str) -> EmbeddingLane:
    lane = resolve_lane_for_doc_model(conn, doc_id, model_name)
    if lane is None:
        raise SystemExit(
            f"No lane for doc_id={doc_id!r} model_name={model_name!r}. "
            "Expected a row in document_lane_index + embedding_lanes."
        )
    return lane


def print_hits(title: str, result: dict) -> None:
    print(f"\n=== {title} ===")
    ids = (result.get("ids") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    for rank, (cid, dist, doc) in enumerate(zip(ids, dists, docs), start=1):
        preview = (doc or "")[:200].replace("\n", " ")
        print(f"  {rank}. id={cid} distance={dist:.4f}")
        print(f"      {preview}...")


def main() -> None:
    pdf = DEFAULT_PDF
    if not pdf.is_file():
        raise SystemExit(f"Missing PDF: {pdf}")

    #init_db(SQLITE_PATH)
    #doc_id = ingest_pdf(pdf)
    doc_id = "doc_AI_Information_en_zh-CN"
    with connect(SQLITE_PATH) as conn:
        rows = fetch_all_chunks(conn, doc_id)
        model_names = [
            r["model_name"]
            for r in conn.execute("SELECT model_name FROM embedding_lanes ORDER BY lane_id")
        ]
    print(
        f"Indexed doc_id={doc_id}, {len(rows)} chunks in SQLite; "
        f"{len(model_names)} lanes in DB."
    )

    question = "什么是监督学习和无监督学习？"
    with connect(SQLITE_PATH) as conn:
        for model_name in model_names:
            lane = require_lane(conn, doc_id, model_name)
            out = query_lane(lane, question, n_results=3)
            print_hits(f"{doc_id} + {model_name} -> lane {lane.lane_id}", out)

    print("\n--- Same question; lane from (doc_id, model_name) ---")
    with connect(SQLITE_PATH) as conn:
        lane_a = require_lane(conn, doc_id, "all-MiniLM-L6-v2")
        lane_b = require_lane(conn, doc_id, "all-mpnet-base-v2")
    print_hits("MiniLM", query_lane(lane_a, question, 2))
    print_hits("MPNet", query_lane(lane_b, question, 2))
    print(
        "\n(Never embed the query with MPNet and search the MiniLM collection, or vice versa — "
        "spaces differ in dimension and geometry.)"
    )


if __name__ == "__main__":
    main()
