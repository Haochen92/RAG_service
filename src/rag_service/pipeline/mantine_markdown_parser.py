import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llama_index.core.bridge.pydantic import ConfigDict
from llama_index.core.schema import TransformComponent, TextNode
from llama_index.core import Document


class MantineMarkdownChunker(TransformComponent):
    # TransformComponent is a Pydantic BaseModel subclass
    # Allow setting custom attributes on this Pydantic-based component
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(
        self,
        chunk_size: int = 3000,
        chunk_overlap: int = 300,
        max_nonempty_header_lines: int = 12,
        inject_context_into_text: bool = True,
        sep_pattern: Optional[Pattern[str]] = None,
        h3_line: Optional[Pattern[str]] = None,
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        super().__init__()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_nonempty_header_lines = max_nonempty_header_lines
        self.inject_context_into_text = inject_context_into_text

        self.sep_pattern = sep_pattern or re.compile(r"(?m)^\-{30,}\s*$")
        self.h3_line = h3_line or re.compile(r"^\s*###\s+(.+?)\s*$")
        self.headers_to_split_on = headers_to_split_on or [("##", "H2"), ("###", "H3")]

        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=True,
            return_each_line=False,
        )

        self._size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", " ", ""],
        )

    def _first_h3_in_header_zone(self, block: str) -> Optional[str]:
        nonempty = 0
        for ln in block.splitlines():
            if ln.strip():
                nonempty += 1
            if nonempty > self.max_nonempty_header_lines:
                break
            m = self.h3_line.match(ln)
            if m:
                return m.group(1).strip()
        return None

    def _chunk_one_document(self, doc: Document) -> List[TextNode]:
        doc_text = doc.get_content()  # safest way across LlamaIndex versions
        base_meta: Dict[str, Any] = dict(doc.metadata or {})
        doc_id = getattr(doc, "doc_id", None) or base_meta.get("doc_id") or "doc"

        raw_blocks = [b.strip() for b in self.sep_pattern.split(doc_text) if b.strip()]
        out: List[TextNode] = []
        chunk_index = 0

        for block in raw_blocks:
            topic = self._first_h3_in_header_zone(block) or "General"

            structural_docs = self._md_splitter.split_text(block)
            sized_docs = self._size_splitter.split_documents(structural_docs)

            for d in sized_docs:
                section = d.metadata.get("H3") or d.metadata.get("H2") or "Overview"
                if section == topic:
                    section = "Overview"

                content = d.page_content
                text_for_embed = (
                    f"Topic: {topic}\nSection: {section}\n\n{content}"
                    if self.inject_context_into_text
                    else content
                )

                # Merge: doc metadata + header metadata + our fields
                meta = {**base_meta, **dict(d.metadata)}
                meta.update(
                    {
                        "document_id": str(doc_id),
                        "chunk_index": chunk_index,
                        "chunk_id": f"{doc_id}::{chunk_index:05d}",
                        "topic": topic,
                        "section": section,
                    }
                )

                node = TextNode(text=text_for_embed, metadata=meta)
                node.excluded_llm_metadata_keys = [
                    "topic",
                    "section",
                    "chunk_id",
                    "chunk_index",
                    "document_id",
                ]
                node.excluded_embed_metadata_keys = [
                    "topic",
                    "section",
                    "chunk_id",
                    "chunk_index",
                    "document_id",
                ]

                out.append(node)
                chunk_index += 1

        return out

    # Make it callable by transformation pipeline
    def __call__(self, documents: List[Document], **kwargs) -> List[TextNode]:
        nodes: List[TextNode] = []
        for doc in documents:
            nodes.extend(self._chunk_one_document(doc))
        return nodes
