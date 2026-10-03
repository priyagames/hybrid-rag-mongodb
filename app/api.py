import os
import shutil
import tempfile
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from app.ingest import ingest_pdf
from app.agents import hybrid_app
from app.baseline import baseline_answer

api = FastAPI(title="Hybrid RAG on MongoDB")


class AskRequest(BaseModel):
    doc_id: str
    question: str


@api.post("/ingest")
def ingest(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        return ingest_pdf(path, file.filename)
    finally:
        os.remove(path)


def _fail(exc):
    code = 429 if "429" in str(exc) else 500
    raise HTTPException(status_code=code, detail=str(exc)[:500])


@api.post("/ask/hybrid")
def ask_hybrid(req: AskRequest):
    try:
        s = hybrid_app.invoke({"question": req.question, "doc_id": req.doc_id})
    except Exception as exc:
        _fail(exc)
    return {"answer": s["answer"], "route": s["route"],
            "route_reason": s.get("route_reason"),
            "table_result": s.get("table_result"),
            "text_result": s.get("text_result", [])}


@api.post("/ask/baseline")
def ask_baseline(req: AskRequest):
    try:
        return baseline_answer(req.question, req.doc_id)
    except Exception as exc:
        _fail(exc)


@api.get("/health")
def health():
    return {"status": "ok"}
