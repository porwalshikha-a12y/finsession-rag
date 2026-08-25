"""FastAPI backend for FinSession-RAG.

Exposes the agent as a small JSON API, with a plain HTML/JS front end
served from static/index.html.

Run from the project root (venv active, index built, OPENAI_API_KEY set):
    pip install fastapi uvicorn
    uvicorn api:app --reload
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import RAGAgent
from src.eviction import make_policy
from src.index import Embedder, make_store
from src.llm import LLM
from src.memory import SemanticMemory

app = FastAPI(title="FinSession-RAG API")

# Loaded once at startup; the current agent (one session at a time).
STATE: dict = {"cfg": None, "embedder": None, "index": None, "agent": None}

MODES = {
    "A0": {"memory": False, "eviction": None,         "bounded": False},
    "A1": {"memory": True,  "eviction": None,         "bounded": False},
    "A2": {"memory": True,  "eviction": "lru",        "bounded": True},
    "A3": {"memory": True,  "eviction": "lfu",        "bounded": True},
    "A4": {"memory": True,  "eviction": "redundancy", "bounded": True},
    "A5": {"memory": True,  "eviction": "cost_aware", "bounded": True},
}


@app.on_event("startup")
def load_resources() -> None:
    cfg = yaml.safe_load(open("config.yaml"))
    embedder = Embedder(**cfg["embeddings"])
    STATE.update(cfg=cfg, embedder=embedder,
                 index=make_store(cfg, embedder))


class SessionRequest(BaseModel):
    mode: str = "A5"          # A0..A5
    budget: int = 20
    tau: float = 0.80
    reuse_check: bool = True


class AskRequest(BaseModel):
    question: str


@app.post("/session")
def new_session(req: SessionRequest) -> dict:
    """Start a fresh session (fresh memory + fresh token ledger)."""
    if req.mode not in MODES:
        raise HTTPException(400, f"unknown mode {req.mode}; use A0..A5")
    cfg, mode = STATE["cfg"], MODES[req.mode]
    memory = None
    if mode["memory"]:
        memory = SemanticMemory(
            embedder=STATE["embedder"],
            eviction_policy=make_policy(mode["eviction"]) if mode["eviction"] else None,
            budget=req.budget if mode["bounded"] else None,
            similarity_threshold=req.tau,
        )
    STATE["agent"] = RAGAgent(
        llm=LLM(**cfg["llm"]),
        index=STATE["index"],
        memory=memory,
        top_k=cfg["retrieval"]["top_k"],
        max_steps=cfg["agent"]["max_steps"],
        reuse_verification=req.reuse_check,
    )
    return {"ok": True, "mode": req.mode}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    agent: RAGAgent | None = STATE["agent"]
    if agent is None:
        raise HTTPException(400, "start a session first (POST /session)")
    trace = agent.answer(req.question)
    mem = agent.memory
    return {
        "answer": trace.answer,
        "trace": {
            "tokens": trace.tokens_spent,
            "retrievals": trace.retrievals,
            "memory_hits": trace.memory_hits,
            "sub_steps": trace.sub_steps,
        },
        "ledger": agent.llm.ledger.snapshot(),
        "memory": None if mem is None else {
            "size": len(mem.entries),
            "budget": mem.budget,
            "stats": mem.stats.snapshot(),
            "entries": [{
                "sub_question": e.sub_question,
                "sub_answer": e.sub_answer,
                "cost_tokens": e.cost_tokens,
                "hits": e.hits,
            } for e in mem.entries],
        },
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
