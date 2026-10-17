from collections.abc import Sequence
from pathlib import Path

import chromadb

from chunks import Chunk
from config import settings


class IngestError(Exception):
    pass


def open_collection() -> chromadb.Collection:
    path = settings.data_dir / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    # Embeddings are computed with Gemini and passed in directly.
    return client.get_or_create_collection(
        name="sources",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


collection = open_collection()


def save_original(doc_id: str, filename: str, data: bytes) -> None:
    folder = settings.data_dir / "uploads" / doc_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_bytes(data)


def add_chunks(doc_id: str, filename: str, chunks: list[Chunk], embeddings: list[Sequence[float]]) -> None:
    stored: list[Sequence[float] | Sequence[int]] = []
    stored.extend(embeddings)
    collection.upsert(
        ids=[f"{doc_id}:{chunk.chunk_index}" for chunk in chunks],
        embeddings=stored,
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": chunk.chunk_index,
                "word_start": chunk.word_start,
            }
            for chunk in chunks
        ],
    )


def search_chunks_by_vector(vector: Sequence[float], top_k: int) -> str:
    query_embeddings: list[Sequence[float] | Sequence[int]] = [vector]
    found = collection.query(query_embeddings=query_embeddings, n_results=top_k)
    # get the first output (we pass only one query embedding)
    documents = (found["documents"] or [[]])[0]
    metadatas = (found["metadatas"] or [[]])[0]
    if not documents:
        return "No uploaded documents matched."

    blocks: list[str] = []
    for document, metadata in zip(documents, metadatas, strict=False):
        filename = metadata.get("filename", "unknown")
        chunk_index = metadata.get("chunk_index", "?")
        blocks.append(f"[{filename} chunk {chunk_index}]\n{document}")
    return "\n\n".join(blocks) if blocks else "No uploaded documents matched."


def decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestError("file is not valid UTF-8 text") from exc


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    if not name or name in {".", ".."}:
        return "upload.txt"
    return name
