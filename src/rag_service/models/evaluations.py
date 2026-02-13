from pydantic import BaseModel, Field


class QueryItem(BaseModel):
    id: str
    category: str
    difficulty: int
    text: str
    tags: list[str] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    query_id: str
    query_text: str
    run_name: str
    param_value: int
    rank: int
    dist: float
    chunk_id: str
    chunk_text: str


class KeywordSearchHit(BaseModel):
    query_id: str
    query_text: str
    run_name: str
    param_value: int
    rank: int
    score: float
    chunk_id: str
    chunk_text: str
