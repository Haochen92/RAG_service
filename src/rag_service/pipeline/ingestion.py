from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import delete

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core import Document as LlamaDocument

from rag_service.models.embeddings import Document, Chunk
import hashlib
import re


SessionFactory = Callable[[], AsyncSession]


class IngestPipeline:
    def __init__(
        self,
        embedding_model: Any,
        chunker_transform: Any,
        session_factory: SessionFactory,
        extra_doc_metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.chunker_transform = chunker_transform
        self.session_factory = session_factory
        self.extra_doc_metadata = extra_doc_metadata or {}

        # LlamaIndex pipeline: chunk -> embed
        self._pipeline = IngestionPipeline(
            transformations=[
                self.chunker_transform,
                self.embedding_model,
            ]
        )

    async def ingest_documents(
        self, documents: List[LlamaDocument], source: str, title: str
    ) -> Dict[str, Any]:
        """
        MVP:
        1) LlamaIndex pipeline (chunk + embed) outside DB transaction
        2) single DB transaction:
           - delete existing documents for source (chunks cascade)
           - insert fresh document
           - bulk insert chunks (no upsert needed because doc_id is new)
        """
        # Transform documents into nodes with embeddings
        nodes: Sequence[BaseNode] = await self._pipeline.arun(
            documents=documents, show_progress=True
        )

        print(f"Ingested {len(nodes)} chunks for source {source}")

        # 2) Store document and chunk
        async with self.session_factory() as session:
            async with session.begin():
                # delete previous docs for this source (chunks cascade)
                await session.execute(delete(Document).where(Document.source == source))

                doc_row = Document(
                    source=source,
                    title=title,
                    embedding_model=getattr(self.embedding_model, "model_name", None),
                    doc_metadata={
                        **self.extra_doc_metadata,
                        "n_nodes": len(nodes),
                    },
                )
                session.add(doc_row)
                await session.flush()  # get doc_row.id populated
                doc_id = doc_row.id

                if nodes:
                    rows = []
                    for i, node in enumerate(nodes):
                        emb = node.get_embedding()
                        if emb is None:
                            raise RuntimeError("Node missing embedding. Did embedding_model run?")

                        content = node.get_content(metadata_mode=MetadataMode.NONE)

                        rows.append(
                            {
                                "document_id": doc_id,
                                "chunk_index": i,
                                "content": content,
                                "embedding": emb,
                                "content_hash": self.create_content_hash(content),
                                "chunk_metadata": dict(node.metadata or {}),
                            }
                        )

                    stmt = pg_insert(Chunk.__table__).on_conflict_do_nothing(
                        index_elements=["document_id", "content_hash"]
                    )
                    await session.execute(stmt, rows)

        return {"document_id": doc_id, "doc_row": doc_row, "n_chunks": len(nodes)}

    def create_content_hash(self, content: str) -> str:
        """Create a simple hash of the content for deduplication purposes."""

        norm = re.sub(r"\s+", " ", str(content)).strip()
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()
