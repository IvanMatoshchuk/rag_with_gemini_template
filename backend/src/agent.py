import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence

from google import genai
from google.genai import types

from chunks import chunk_text
from config import settings
from db import insert_document, mark_failed, mark_ready
from vector_store import IngestError, add_chunks, decode_text, safe_filename, save_original, search_chunks_by_vector

MAX_TURNS = 5
EMBED_BATCH = 100
EMBED_WORKERS = 4

SYSTEM_PROMPT = """
You are a senior research assistant.
Use search_documents to look through the user's uploaded sources before you answer.
If nothing relevant was uploaded, say so.

*IMPORTANT*:
- DO NOT INVENT sources. Write the answer in markdown.
- ALWAYS CITE the filename and chunk index from the tool results!
- Provide citations in the format of [filename:chunk_index].
"""

WEB_SEARCH_PROMPT = """
Google Search is available. Use it for facts that are not in the uploaded files.
Cite web pages using the format [title](url).
"""

SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description="Search the user's uploaded source documents.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for in the uploaded documents.",
                    }
                },
                "required": ["query"],
            },
        )
    ]
)


class ResearchAgent:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self._embed_slots = asyncio.Semaphore(EMBED_WORKERS)

    def tools(self) -> list[types.Tool]:
        tools = [SEARCH_TOOL]
        if settings.google_search_enabled:
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        return tools

    def _tool_config(self) -> types.ToolConfig | None:
        if not settings.google_search_enabled:
            return None
        return types.ToolConfig(include_server_side_tool_invocations=True)

    async def embed_texts(self, texts: list[str], task_type: str) -> list[Sequence[float]]:
        batches = [texts[start : start + EMBED_BATCH] for start in range(0, len(texts), EMBED_BATCH)]
        embedded = await asyncio.gather(*(self._embed_batch(batch, task_type) for batch in batches))
        return [vector for batch_vectors in embedded for vector in batch_vectors]

    async def _embed_batch(self, batch: list[str], task_type: str) -> list[Sequence[float]]:
        async with self._embed_slots:
            response = await self.client.aio.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=batch,  # type: ignore[arg-type]
                config=types.EmbedContentConfig(task_type=task_type),
            )
        return self._read_embeddings(response, len(batch))

    def _read_embeddings(self, response: types.EmbedContentResponse, expected: int) -> list[Sequence[float]]:
        embeddings = response.embeddings or []
        if len(embeddings) != expected:
            raise RuntimeError("embedding response did not match the input batch")
        vectors: list[Sequence[float]] = []
        for embedding in embeddings:
            values = embedding.values
            if not values:
                raise RuntimeError("embedding response was empty")
            vectors.append(values)
        return vectors

    async def ingest_file(self, filename: str, content_type: str, data: bytes) -> tuple[str, int, str]:
        # TODO: Use hashing to check whether uploaded data already exists
        safe_name = safe_filename(filename)
        doc_id = uuid.uuid4().hex
        insert_document(doc_id, safe_name)
        try:
            text = decode_text(data)
            chunks = chunk_text(text, settings.chunk_words, settings.chunk_overlap)
            if not chunks:
                raise IngestError(f"{safe_name} has no text")
            save_original(doc_id, safe_name, data)
            embeddings = await self.embed_texts([chunk.text for chunk in chunks], "RETRIEVAL_DOCUMENT")
            add_chunks(doc_id, safe_name, chunks, embeddings)
        except IngestError as exc:
            mark_failed(doc_id, str(exc))
            raise
        except Exception as exc:
            mark_failed(doc_id, str(exc))
            raise

        mark_ready(doc_id)
        return safe_name, len(data), content_type

    async def search_documents(self, query: str) -> str:
        cleaned = query.strip()
        if not cleaned:
            return "No query was provided."
        vector = (await self.embed_texts([cleaned], "RETRIEVAL_QUERY"))[0]
        return search_chunks_by_vector(vector, settings.top_k)

    async def stream_research(self, question: str) -> AsyncIterator[str]:
        try:
            async for piece in self._answer(question):
                yield piece
        except Exception as exc:
            yield f"\n\nResearch failed: {exc}\n"

    async def _answer(self, question: str) -> AsyncIterator[str]:
        contents: list[types.Content] = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        prompt = SYSTEM_PROMPT + (WEB_SEARCH_PROMPT if settings.google_search_enabled else "")
        config = types.GenerateContentConfig(
            system_instruction=prompt,
            tools=self.tools(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=self._tool_config(),
        )

        for _ in range(MAX_TURNS):
            parts: list[types.Part] = []
            calls: list[types.FunctionCall] = []
            web_sources: list[str] = []
            wrote = False
            stream = await self.client.aio.models.generate_content_stream(
                model=settings.gemini_model,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                candidate = chunk.candidates[0] if chunk.candidates else None
                if candidate is None:
                    continue
                self._collect_web_sources(candidate, found=web_sources)
                content = candidate.content
                if content is None or not content.parts:
                    continue
                for part in content.parts:
                    parts.append(part)
                    call = part.function_call
                    if call is not None and call.name == "search_documents" and not call.will_continue:
                        calls.append(call)
                    elif part.text and not part.thought:
                        wrote = True
                        yield part.text

            if not calls:
                if web_sources:
                    yield "\n\n**Web sources**\n" + "\n".join(web_sources) + "\n"
                if not wrote:
                    yield "I couldn't find an answer in the uploaded sources."
                return

            contents.append(types.Content(role="model", parts=parts))
            tool_parts = [await self._tool_result(call) for call in calls]
            contents.append(types.Content(role="user", parts=tool_parts))

        yield "I couldn't finish that research request."

    async def _tool_result(self, call: types.FunctionCall) -> types.Part:
        return types.Part.from_function_response(
            name="search_documents",
            response={"result": await self.search_documents(self._query(call))},
        )

    def _collect_web_sources(self, candidate: types.Candidate, found: list[str]) -> None:
        metadata = candidate.grounding_metadata
        if metadata is None or not metadata.grounding_chunks:
            return
        for chunk in metadata.grounding_chunks:
            web = chunk.web
            if web is None or not web.uri:
                continue
            line = f"- [{web.title or web.uri}]({web.uri})"
            if line not in found:
                found.append(line)

    def _query(self, call: types.FunctionCall) -> str:
        if call.args is None:
            return ""
        raw = call.args.get("query", "")
        if not isinstance(raw, str):
            return ""
        return raw.strip()


agent = ResearchAgent()
