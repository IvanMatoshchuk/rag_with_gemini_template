import uuid
from collections.abc import Iterator, Sequence

from google import genai
from google.genai import types

from chunks import chunk_text
from config import settings
from db import insert_document, mark_failed, mark_ready
from vector_store import IngestError, add_chunks, decode_text, safe_filename, save_original, search_chunks_by_vector

MAX_TURNS = 5
EMBED_BATCH = 100

SYSTEM_PROMPT = """
You are a senior research assistant.
Use search_documents to look through the user's uploaded sources before you answer.
If nothing relevant was uploaded, say so.

*IMPORTANT*:
- DO NOT INVENT sources. Write the answer in markdown.
- ALWAYS CITE the filename and chunk index from the tool results! 
- Provide citations in the format of [filename:chunk_index].
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

    def embed_texts(self, texts: list[str], task_type: str) -> list[Sequence[float]]:
        vectors: list[Sequence[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            response = self.client.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise RuntimeError("embedding response did not match the input batch")
            for embedding in embeddings:
                values = embedding.values
                if not values:
                    raise RuntimeError("embedding response was empty")
                vectors.append(values)
        return vectors

    def ingest_file(self, filename: str, content_type: str, data: bytes) -> tuple[str, int, str]:
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
            embeddings = self.embed_texts([chunk.text for chunk in chunks], "RETRIEVAL_DOCUMENT")
            add_chunks(doc_id, safe_name, chunks, embeddings)
        except IngestError as exc:
            mark_failed(doc_id, str(exc))
            raise
        except Exception as exc:
            mark_failed(doc_id, str(exc))
            raise

        mark_ready(doc_id)
        return safe_name, len(data), content_type

    def search_documents(self, query: str) -> str:
        cleaned = query.strip()
        if not cleaned:
            return "No query was provided."
        vector = self.embed_texts([cleaned], "RETRIEVAL_QUERY")[0]
        return search_chunks_by_vector(vector, settings.top_k)

    def stream_research(self, question: str) -> Iterator[str]:
        try:
            yield from self._answer(question)
        except Exception as exc:
            yield f"\n\nResearch failed: {exc}\n"

    def _answer(self, question: str) -> Iterator[str]:
        contents: list[types.Content] = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        for _ in range(MAX_TURNS):
            parts: list[types.Part] = []
            calls: list[types.FunctionCall] = []
            wrote = False
            for chunk in self.client.models.generate_content_stream(
                model=settings.gemini_model,
                contents=contents,
                config=config,
            ):
                content = chunk.candidates[0].content if chunk.candidates else None
                if content is None or not content.parts:
                    continue
                for part in content.parts:
                    print("\n\n******", part)
                    parts.append(part)
                    call = part.function_call
                    if call is not None and not call.will_continue:
                        calls.append(call)
                    elif part.text and not part.thought:
                        wrote = True
                        yield part.text

            if not calls:
                if not wrote:
                    yield "I couldn't find an answer in the uploaded sources."
                return

            contents.append(types.Content(role="model", parts=parts))
            contents.append(types.Content(role="user", parts=[self._tool_result(call) for call in calls]))

        yield "I couldn't finish that research request."

    def _tool_result(self, call: types.FunctionCall) -> types.Part:
        name = call.name or "search_documents"
        if name != "search_documents":
            response: dict[str, str] = {"error": f"Unknown tool {name}"}
        else:
            response = {"result": self.search_documents(self._query(call))}
        return types.Part.from_function_response(name=name, response=response)

    def _query(self, call: types.FunctionCall) -> str:
        if call.args is None:
            return ""
        raw = call.args.get("query", "")
        if not isinstance(raw, str):
            return ""
        return raw.strip()


agent = ResearchAgent()
