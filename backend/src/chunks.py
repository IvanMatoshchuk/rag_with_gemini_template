from pydantic import BaseModel


class Chunk(BaseModel):
    text: str
    chunk_index: int
    word_start: int


def chunk_text(text: str, chunk_words: int, overlap: float) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    step = max(int(chunk_words * (1 - overlap)), 1)
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        piece = words[start : start + chunk_words]
        chunks.append(Chunk(text=" ".join(piece), chunk_index=index, word_start=start))
        if start + chunk_words >= len(words):
            break
        start += step
        index += 1
    return chunks
