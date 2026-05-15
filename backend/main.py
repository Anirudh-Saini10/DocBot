import os
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agent import run_agent
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DocBot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    role: str
    content: str
    tool_calls: List[Dict[str, Any]] = []


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]
    result = run_agent(req.message, conversation_history=history)
    return ChatResponse(
        role=result["role"],
        content=result["content"],
        tool_calls=result.get("tool_calls", []),
    )
