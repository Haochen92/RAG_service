import numpy as np
import pandas as pd
from .eval_utils import prepare_ranked_list


def dcg_at_k(relevance: np.ndarray, k: int) -> float:
    rels = np.asarray(relevance, dtype=float)[:k]
    if rels.size == 0:
        return 0.0

    # Calculate gains
    gains = (2**rels) - 1

    # Calculate discounts
    discounts = np.log2(np.arange(2, rels.size + 2))

    score = (gains / discounts).sum()
    return score


def calculate_ndcg(df: pd.DataFrame, k: int) -> list[dict[str, float]]:
    """
    Calculate the ideal DCG (IDCG) for the given DataFrame.
    """
    df = prepare_ranked_list(df)
    rows = []
    for query_id, group in df.groupby("query_id"):
        rels = group["relevance"].to_numpy()
        rels_sorted_desc = np.sort(rels)[::-1]

        dcg_value = dcg_at_k(rels, k)
        idcg_value = dcg_at_k(rels_sorted_desc, k)

        ndcg_value = dcg_value / idcg_value if idcg_value > 0 else 0.0
        rows.append({"query_id": query_id, "ndcg_value": ndcg_value, "dcg_value": dcg_value})

    return rows
