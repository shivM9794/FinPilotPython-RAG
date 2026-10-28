
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
from uuid import uuid4

from app.graph.rag_graph import rag_graph


# -------------------------
# FastAPI Application
# -------------------------
app = FastAPI(
    title="FinPilot AI",
    description="Finance RAG application",
    version="1.0.0",
)


# -------------------------
# Temporary In-Memory Storage
# Documents are lost when the server restarts.
# -------------------------
documents = {}


# -------------------------
# HTML User Interface
# -------------------------
HTML_FILE = Path(__file__).parent / "templates" / "index.html"


# -------------------------
# Request Models
# -------------------------
class SearchRequest(BaseModel):
    user_id: str
    question: str
    top_k: int = 5


# -------------------------
# Web UI
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    if not HTML_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="UI file not found. Create app/templates/index.html",
        )

    return HTML_FILE.read_text(encoding="utf-8")


# -------------------------
# Health Check
# -------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# -------------------------
# Document Ingestion
# -------------------------
@app.post("/api/v1/knowledge/ingest")
async def ingest_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    filename = file.filename or "document.txt"

    if Path(filename).suffix.lower() != ".txt":
        raise HTTPException(
            status_code=400,
            detail="Currently only .txt files are supported",
        )

    content = await file.read()

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must contain UTF-8 text",
        )
    finally:
        await file.close()

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="File is empty",
        )

    document_id = str(uuid4())

    documents[document_id] = {
        "document_id": document_id,
        "user_id": user_id,
        "filename": filename,
        "text": text,
    }

    return {
        "document_id": document_id,
        "filename": filename,
        "pages_processed": 1,
        "chunks_created": 1,
        "collection": "finpilot_knowledge",
        "message": "Document uploaded successfully",
    }


# -------------------------
# Document Search with LangGraph
# -------------------------
@app.post("/api/v1/knowledge/search")
def search_documents(request: SearchRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    if request.top_k < 1 or request.top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20",
        )

    result = rag_graph.invoke({
        "user_id": request.user_id,
        "question": request.question,
        "top_k": request.top_k,
        "documents": list(documents.values()),
        "results": [],
    })

    return {
        "question": request.question,
        "results": result["results"],
        "message": (
            "Search completed using LangGraph. "
            "Keyword retrieval is enabled; AI-generated "
            "answers are not enabled yet."
        ),
    }
