from sqlalchemy import select, text
from ..models import QueryItem, RetrievalHit, KeywordSearchHit, Document, Chunk
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import asyncio


async def vectors_search(
    *,
    queries: list[QueryItem],
    source: str,
    embedding_model,
    ef_search_values: list[int],
    k: int = 15,
    session: AsyncSession,
) -> list[RetrievalHit]:
    query_embeds = {q.id: embedding_model.get_query_embedding(q.text) for q in queries}

    hits: list[RetrievalHit] = []

    for ef in ef_search_values:
        run_name = f"hnsw_ef{ef}_k{k}"

        for q in queries:
            q_emb = query_embeds[q.id]
            dist = Chunk.embedding.cosine_distance(q_emb).label("dist")

            stmt = (
                select(Chunk.id.label("chunk_id"), Chunk.content.label("chunk_text"), dist)
                .join(Document, Document.id == Chunk.document_id)
                .where(Document.source == source)  # or .where(Chunk.document_id == doc_id)
                .order_by(dist)
                .limit(k)
            )

            async with session.begin():  # needed for SET LOCAL
                await session.execute(
                    text(f"SET LOCAL hnsw.ef_search = {ef}"),
                )
                rows = (await session.execute(stmt)).mappings().all()

            for rank, r in enumerate(rows, start=1):
                hits.append(
                    RetrievalHit(
                        query_id=q.id,
                        query_text=q.text,
                        run_name=run_name,
                        param_value=ef,
                        rank=rank,
                        dist=float(r["dist"]),
                        chunk_id=str(r["chunk_id"]),
                        chunk_text=r["chunk_text"],
                    )
                )

    return hits


async def bm25_search(
    *,
    queries: list[QueryItem],
    k: int = 15,
    session: AsyncSession,
    source: str,
) -> list[KeywordSearchHit]:
    hits: list[KeywordSearchHit] = []

    sql = text(
        """
        SELECT
            c.id AS chunk_id,
            c.content AS chunk_text,
            pdb.score(c.id) AS score
        FROM chunks AS c
        JOIN documents AS d ON d.id = c.document_id
        WHERE d.source = :source
          AND c.content ||| :q
        ORDER BY score DESC
        LIMIT :lim
    """
    )

    for q in queries:
        kw_res = await session.execute(sql, {"q": q.text, "lim": k, "source": source})
        rows = kw_res.mappings().all()

        for rank, r in enumerate(rows, start=1):
            hits.append(
                KeywordSearchHit(
                    query_id=q.id,
                    query_text=q.text,
                    run_name="bm25",
                    param_value=k,
                    rank=rank,
                    score=float(r["score"]),
                    chunk_id=str(r["chunk_id"]),
                    chunk_text=r["chunk_text"],
                )
            )

    return hits


async def hybrid_search(
    *,
    queries: list[QueryItem],
    source: str,
    embedding_model,
    ef_search_values: list[int],
    k: int = 15,
    rrf_k: int = 60,
    a: float = 0.5,
    b: float = 0.5,
    session: AsyncSession,
) -> pd.DataFrame:

    vector_hits = await vectors_search(
        queries=queries,
        source=source,
        embedding_model=embedding_model,
        ef_search_values=ef_search_values,
        k=k,
        session=session,
    )
    keyword_hits = await bm25_search(
        queries=queries,
        k=k,
        session=session,
        source=source,
    )

    vector_df = pd.DataFrame([hit.model_dump() for hit in vector_hits])
    keyword_df = pd.DataFrame([hit.model_dump() for hit in keyword_hits])

    hybrid_search_results = calculate_rrf_rank(
        vector_df=vector_df,
        keyword_df=keyword_df,
        rrf_k=rrf_k,
        a=a,
        b=b,
    )

    return hybrid_search_results


def calculate_rrf_rank(
    vector_df: pd.DataFrame,
    keyword_df: pd.DataFrame,
    rrf_k: int = 60,
    a: float = 0.5,
    b: float = 0.5,
) -> pd.DataFrame:
    """
    Calculate RRF (Reciprocal Rank Fusion) combined ranks.

    Parameters:
        vector_array: np.ndarray of shape (n_queries, n_docs) with vector search ranks
        bm25_array: np.ndarray of shape (n_queries, n_docs) with BM25 search ranks
        rrf_k: int, the RRF constant to use in the formula
        a: float, weight for vector search scores
        b: float, weight for keyword search scores
    Returns:
        np.ndarray of shape (n_queries, n_docs) re-ranked using combined RRF scores
    """

    v = vector_df.copy()
    k = keyword_df.copy()

    # Ensure ranks are numeric
    v["rank"] = pd.to_numeric(v["rank"], errors="coerce")
    k["rank"] = pd.to_numeric(k["rank"], errors="coerce")

    # Calculate scores with vectorized operations
    v["score"] = a / (rrf_k + v["rank"])
    k["score"] = b / (rrf_k + k["rank"])

    # Combine both tables
    fused = pd.concat([v, k], ignore_index=True)

    # Sum scores
    fused = fused.groupby(["query_id", "query_text", "chunk_id", "chunk_text"], as_index=False)[
        "score"
    ].sum()

    # Sort within each query and assign new ranks
    fused = fused.drop_duplicates(subset=["query_id", "chunk_id"], keep="first")
    fused = fused.sort_values(["query_id", "score"], ascending=[True, False]).reset_index(drop=True)
    fused["rank"] = fused.groupby("query_id").cumcount() + 1

    return fused
