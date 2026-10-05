"""HTTP API. Also serves the built React app, so one service is the whole demo.

    uvicorn backend.app.main:app --reload
"""
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .agents import run_audit
from .corpus import load_corpus
from .llm import describe, get_chat_model
from .retrieval import get_retriever

app = FastAPI(title="Compliance Audit Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AuditRequest(BaseModel):
    text: str = Field(min_length=40, max_length=config.MAX_PROPOSAL_CHARS)
    title: str = Field("Proposal", max_length=120)


def _samples():
    out = []
    for path in sorted(config.PROPOSAL_DIR.glob("*.txt")):
        text = path.read_text()
        out.append({"id": path.stem, "title": text.splitlines()[0].strip(), "text": text})
    return out


@app.get("/api/health")
def health():
    corpus = load_corpus()
    return {"status": "ok", "passages": len(corpus),
            "sections": len({p.section for p in corpus}),
            "parts": sorted({p.part for p in corpus}, key=int),
            "analyst": describe(get_chat_model())}


@app.get("/api/samples")
def samples():
    return _samples()


@app.post("/api/audit")
def audit(req: AuditRequest):
    result = run_audit(req.text, get_retriever(), model=get_chat_model(), title=req.title)
    if not result["findings"]:
        raise HTTPException(422, "No clauses found. Write each commitment as its own sentence or numbered item.")
    return result


@app.get("/api/passages/{passage_id:path}")
def passage(passage_id: str):
    for p in load_corpus():
        if p.id == passage_id:
            return asdict(p)
    raise HTTPException(404, "Unknown passage id")


if config.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=config.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        return FileResponse(config.FRONTEND_DIST / "index.html")
