# Research assistant backend

Project-wide setup, architecture, examples, and evaluation: [../README.md](../README.md).

FastAPI-based backend for the research-agent UI. It stores uploaded texts (pdfs not suppported yet), chunks and embeds them, and streams a markdown answer that can search those chunks.

## Setup

From the repository root, copy env defaults and add your Gemini API key:

```sh
cp .env.example .env
make run-backend-local
```


Environment variables are read only from the top-level `.env` (not `backend/.env`). The model, embedding model, chunk size, and overlap all have defaults in `.env.example`.

The API listens on `http://localhost:8787`. 
Check frontend instructions in [README.md](../frontend/README.md)

Run backend with frontend with:
```sh
make run-local
```

Uploaded files, the sqlite database, and the Chroma index are written under `data/` (gitignored). That directory is what a later Docker volume would mount.

## Endpoints

`POST /api/sources` accepts multipart field `files`. Several files in one request are ingested together with `asyncio.gather`. Each upload is stored as a new document: the original file, a sqlite row, word chunks, and embeddings. Embedding batches for a file are gathered the same way, capped at four at a time. The response is `{ "uploaded": [{ "name", "size", "type" }] }`. A file that cannot be read is marked `failed` and the error text is returned.

`POST /api/research` accepts `{ "request": "..." }` and streams markdown as plain text chunks. `ResearchAgent` holds one Gemini client, used both to embed and to generate. It can call `search_documents`, which embeds the query and reads the nearest chunks from Chroma. When `GOOGLE_SEARCH_ENABLED` is true (the default), the model can also ground the answer with Google Search. Web links from that grounding are appended at the end of the answer. Uploaded sources are still cited as `[filename:chunk_index]`.


Run the tests from `backend/` with `uv run pytest` or `make test` from the project's root.
