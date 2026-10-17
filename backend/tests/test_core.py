import asyncio
from types import SimpleNamespace

from chunks import chunk_text


def test_chunk_text_overlaps_and_skips_blank_input() -> None:
    chunks = chunk_text(" ".join(f"w{i}" for i in range(40)), chunk_words=20, overlap=0.15)

    assert [chunk.word_start for chunk in chunks] == [0, 17, 34]
    assert chunks[0].chunk_index == 0
    assert chunks[-1].text.endswith("w39")
    assert chunk_text("   ", chunk_words=10, overlap=0.15) == []


def test_google_search_tool_follows_the_setting(monkeypatch) -> None:
    from agent import ResearchAgent
    from config import settings

    monkeypatch.setattr(settings, "google_search_enabled", True)
    enabled = ResearchAgent().tools()
    assert any(tool.google_search is not None for tool in enabled)

    monkeypatch.setattr(settings, "google_search_enabled", False)
    disabled = ResearchAgent().tools()
    assert all(tool.google_search is None for tool in disabled)
    assert disabled[0].function_declarations[0].name == "search_documents"


def test_uploaded_file_can_be_searched(monkeypatch) -> None:
    from agent import agent

    async def fake_embed(texts: list[str], _task: str) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def run() -> tuple[str, int, str]:
        name, size, _content_type = await agent.ingest_file("notes.txt", "text/plain", b"refund window is 30 days")
        found = await agent.search_documents("refund window")
        return name, size, found

    monkeypatch.setattr(agent, "embed_texts", fake_embed)
    name, size, found = asyncio.run(run())

    assert name == "notes.txt"
    assert size > 0
    assert "notes.txt chunk 0" in found


def test_embedding_batches_run_together_and_keep_order(monkeypatch) -> None:
    from agent import EMBED_BATCH, agent

    async def run() -> list[list[float]]:
        active = {"count": 0, "max": 0}

        async def fake_embed(*, model: str, contents: list[str], config: object) -> SimpleNamespace:
            del model, config
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await asyncio.sleep(0.01)
            active["count"] -= 1
            embeddings = [SimpleNamespace(values=[float(text)]) for text in contents]
            return SimpleNamespace(embeddings=embeddings)

        monkeypatch.setattr(agent.client.aio.models, "embed_content", fake_embed)
        texts = [str(i) for i in range(EMBED_BATCH * 2)]
        vectors = await agent.embed_texts(texts, "RETRIEVAL_DOCUMENT")
        assert active["max"] == 2
        return vectors

    vectors = asyncio.run(run())
    assert vectors[0] == [0.0]
    assert vectors[-1] == [float(len(vectors) - 1)]


def test_several_uploads_are_ingested_together(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from agent import agent
    from main import app

    active = {"count": 0, "max": 0}

    async def slow_ingest(filename: str, content_type: str, data: bytes) -> tuple[str, int, str]:
        active["count"] += 1
        active["max"] = max(active["max"], active["count"])
        await asyncio.sleep(0.01)
        active["count"] -= 1
        return filename, len(data), content_type

    monkeypatch.setattr(agent, "ingest_file", slow_ingest)
    client = TestClient(app)

    response = client.post(
        "/api/sources",
        files=[
            ("files", ("one.txt", b"alpha", "text/plain")),
            ("files", ("two.txt", b"beta", "text/plain")),
        ],
    )

    assert response.status_code == 200
    assert active["max"] == 2
    assert response.json()["uploaded"] == [
        {"name": "one.txt", "size": 5, "type": "text/plain"},
        {"name": "two.txt", "size": 4, "type": "text/plain"},
    ]
