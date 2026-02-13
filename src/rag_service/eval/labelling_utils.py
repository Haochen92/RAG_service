from ..models import RetrievalHit
from pathlib import Path
from typing import Any
import csv
import pandas as pd


def export_hits_to_csv(hits: list[Any], out_path: str) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for h in hits:
        d = h.model_dump()
        d["relevance"] = ""
        rows.append(d)

    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return str(path)


def dedupe_vector_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    best: dict[tuple[str, str], RetrievalHit] = {}
    for h in hits:
        key = (h.query_id, str(h.chunk_id))
        if key not in best or h.dist < best[key].dist:
            best[key] = h
    return list(best.values())


def make_truth_label_df(labeled_df: pd.DataFrame) -> pd.DataFrame:
    truth = labeled_df[["query_id", "chunk_id", "relevance"]].copy()

    # ensure numeric + handle bad values safely
    truth["relevance"] = pd.to_numeric(truth["relevance"], errors="coerce").fillna(0).astype(int)

    # if duplicates exist, choose a policy:
    # - max(): safest if you might have inconsistent duplicates
    truth = truth.groupby(["query_id", "chunk_id"], as_index=False)["relevance"].max()

    return truth
