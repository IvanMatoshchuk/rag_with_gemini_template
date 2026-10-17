import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from agent import agent
from vector_store import IngestError

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    request: str


class UploadedFile(BaseModel):
    name: str
    size: int
    type: str


class UploadResult(BaseModel):
    uploaded: list[UploadedFile]

# Exception handlers
@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException) -> PlainTextResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return PlainTextResponse(detail, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> PlainTextResponse:
    return PlainTextResponse(str(exc), status_code=422)


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, exc: Exception) -> PlainTextResponse:
    return PlainTextResponse(str(exc), status_code=500)

# API endpoints
@app.post("/api/sources")
def upload_sources(files: list[UploadFile] = File(...)) -> UploadResult:
    uploaded: list[UploadedFile] = []
    for file in files:
        data = file.file.read()
        try:
            name, size, content_type = agent.ingest_file(
                file.filename or "upload.txt",
                file.content_type or "text/plain",
                data,
            )
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        uploaded.append(UploadedFile(name=name, size=size, type=content_type))
    return UploadResult(uploaded=uploaded)


@app.post("/api/research")
def research(body: ResearchRequest) -> StreamingResponse:
    question = body.request.strip()
    if not question:
        raise HTTPException(status_code=400, detail="request is empty")
    return StreamingResponse(agent.stream_research(question), media_type="text/plain")


def main() -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=8787)


if __name__ == "__main__":
    main()
