import os
import pypdf
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ── Gemini configuration ──────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("WARNING: GEMINI_API_KEY is not set. API calls will fail.")


# ── Text utilities ────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def load_pdf_chunks(pdf_path: str) -> list[dict]:
    reader = pypdf.PdfReader(pdf_path)
    chunks = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        for chunk in chunk_text(text):
            chunks.append({"page": idx + 1, "text": chunk.strip()})
    return chunks


# ── RAG system ────────────────────────────────────────────────────────────────
class RAGSystem:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def load(self):
        """Called once at startup — separated so FastAPI can handle errors cleanly."""
        print(f"Loading and indexing PDF: {self.pdf_path}...")
        self.chunks = load_pdf_chunks(self.pdf_path)

        embeddings = []
        for chunk in self.chunks:
            res = genai.embed_content(
                model="models/gemini-embedding-001",
                content=chunk["text"],
                task_type="retrieval_document",
            )
            embeddings.append(res["embedding"])

        self.embeddings = np.array(embeddings)
        print(f"Indexed {len(self.chunks)} chunks successfully.")

    @property
    def ready(self) -> bool:
        return self.embeddings is not None and len(self.chunks) > 0

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query",
        )
        query_emb = np.array(res["embedding"])
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        similarities = np.dot(self.embeddings, query_emb) / (norms * query_norm + 1e-10)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.chunks[i]["text"] for i in top_indices]

    def answer_query(self, query: str) -> str:
        context_str = "\n---\n".join(self.retrieve(query))
        prompt = f"""
You are the AI Assistant on Mohammed Naseef's portfolio website. Answer the user's \
questions about Mohammed Naseef using the resume context below.

Context from resume:
{context_str}

User's Question:
{query}

Instructions:
1. Give a professional, engaging, and accurate answer based on the provided context.
2. If the answer is not in the context, respond helpfully using general knowledge \
   while noting it isn't explicitly in his resume, or invite the user to contact \
   Mohammed Naseef directly.
3. Keep the response concise (1–3 paragraphs). Use clean GitHub-flavored Markdown \
   (bold, bullet points) — no raw HTML tags.
"""
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        return model.generate_content(prompt).text


# ── App lifecycle ─────────────────────────────────────────────────────────────
PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test11.pdf")
rag_system = RAGSystem(PDF_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG index once when the server starts."""
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY missing — RAG will not function.")
    elif not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {PDF_PATH} — RAG will not function.")
    else:
        try:
            rag_system.load()
        except Exception as e:
            print(f"ERROR during RAG initialization: {e}")
    yield  # Server runs here
    # (cleanup goes here if needed)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
async def health():
    """Keep-alive endpoint + readiness check."""
    return {"status": "ok", "rag_ready": rag_system.ready}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not rag_system.ready:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready. Check server logs.",
        )
    try:
        answer = rag_system.answer_query(request.message)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Static files (serves index.html + assets) — must be LAST ─────────────────
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")