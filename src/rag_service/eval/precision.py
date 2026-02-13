import numpy as np
import pandas as pd
from .eval_utils import prepare_ranked_list


def calculate_precision_scores(df: pd.DataFrame, k: int) -> list[dict[str, float]]:
    df = prepare_ranked_list(df)
    rows = []
    for query_id, group in df.groupby("query_id"):
        rels = group["relevance"].to_numpy()

        precision_at_k = calculate_precision_at_K(rels, k)
        average_precision = calculate_average_precision(rels, k)

        rows.append(
            {
                "query_id": query_id,
                "precision_at_k": precision_at_k,
                "average_precision": average_precision,
            }
        )

    return rows


def calculate_precision_at_K(relevance: np.ndarray, k: int) -> float:
    rels = np.asarray(relevance, dtype=float)[:k]
    if rels.size == 0:
        return 0.0
    precision = np.sum(rels > 1) / k
    return precision


def calculate_average_precision(relevance: np.ndarray, k: int) -> float:
    rels = np.asarray(relevance, dtype=float)[:k]
    if rels.size == 0:
        return 0.0
    total_rel = 0
    sum_precisions = 0.0
    for i in range(len(rels)):
        if rels[i] > 1:
            total_rel += 1
            precision_at_i = total_rel / (i + 1)
            sum_precisions += precision_at_i
    average_precision = sum_precisions / total_rel if total_rel > 0 else 0.0
    return average_precision
