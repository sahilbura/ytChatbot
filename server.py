from threading import Lock
from typing import Any
from uuid import uuid4
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from embeddings_groq import GroqEmbeddings
from groq_chat import GroqChat
from pydantic import BaseModel, Field

from chatbot import build_chat_chain, get_translated_transcript


load_dotenv()

app = FastAPI(title="ytChatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = GroqChat()
except Exception:
    model = ChatOpenAI(model="gpt-4o-mini")

try:
    embeddings = GroqEmbeddings()
except Exception:
    embeddings = None

sessions: dict[str, dict[str, Any]] = {}
sessions_lock = Lock()


class SessionCreateRequest(BaseModel):
    video_url: str = Field(min_length=1)


class SessionCreateResponse(BaseModel):
    session_id: str
    video_id: str
    message: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str


def extract_video_id(video_url: str) -> str | None:
    candidate = video_url.strip()
    if not candidate:
        return None

    normalized = candidate
    if "://" not in candidate and candidate.startswith(("www.", "youtube.", "m.youtube.", "youtu.be/")):
        normalized = f"https://{candidate}"

    parsed_url = urlparse(normalized)

    if parsed_url.scheme or parsed_url.netloc:
        if "youtu.be" in parsed_url.netloc:
            short_id = parsed_url.path.lstrip("/").split("/")[0]
            return short_id or None

        query_id = parse_qs(parsed_url.query).get("v")
        if query_id:
            return query_id[0]

        path_parts = [segment for segment in parsed_url.path.split("/") if segment]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            return path_parts[1]

    if "/" not in candidate:
        return candidate

    return None


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/sessions", response_model=SessionCreateResponse)
def create_session(payload: SessionCreateRequest) -> SessionCreateResponse:
    video_id = extract_video_id(payload.video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    transcript = get_translated_transcript(video_id, model)
    if not transcript or transcript == "error":
        raise HTTPException(status_code=400, detail="Could not fetch transcript")

    try:
        chain = build_chat_chain(transcript, model, embeddings)
    except Exception:
        # If embedding setup fails at runtime, continue without vector embeddings.
        chain = build_chat_chain(transcript, model, None)
    session_id = uuid4().hex

    with sessions_lock:
        sessions[session_id] = {"video_id": video_id, "chain": chain}

    return SessionCreateResponse(
        session_id=session_id,
        video_id=video_id,
        message="Transcript loaded successfully",
    )


@app.post("/api/sessions/{session_id}/messages", response_model=ChatResponse)
def create_message(session_id: str, payload: ChatRequest) -> ChatResponse:
    with sessions_lock:
        session = sessions.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    answer = session["chain"].invoke(payload.question)
    return ChatResponse(answer=answer)