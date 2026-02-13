import pandas as pd


# prepare ranked list
def prepare_ranked_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the ranked list by filling missing relevance values,
    sorting and deduplicating.
    """
    df = df.copy()

    df["relevance"] = df["relevance"].fillna(0).astype(int)
    df = df.sort_values(by=["query_id", "rank"], ascending=[True, True])
    df = df.drop_duplicates(subset=["query_id", "chunk_id"], keep="first")
    # keep ranking continguous after deduplication
    df["rank"] = df.groupby("query_id").cumcount() + 1

    return df
