from pathlib import Path
from llama_index.core import SimpleDirectoryReader


def load_corpus(root: Path, source: str, required_exts=None, recursive: bool = True):
    root = root.resolve()
    required_exts = required_exts or [".md", ".txt", ".pdf"]

    def meta(file_path: str) -> dict:
        p = Path(file_path).resolve()
        rel = p.relative_to(root).as_posix()
        return {"source": source, "relative_path": rel, "doc_id": f"{source}::{rel}"}

    docs = SimpleDirectoryReader(
        input_dir=str(root),
        recursive=recursive,
        required_exts=required_exts,
        file_metadata=meta,
    ).load_data()

    for d in docs:
        d.doc_id = d.metadata["doc_id"]

    return docs
