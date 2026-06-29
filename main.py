import os
import pypdf
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

API_KEY = os.environ.get("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("WARNING: GEMINI_API_KEY environment variable is not set. Gemini API calls will fail.")


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_pdf_chunks(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    chunks = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        page_chunks = chunk_text(text)
        for chunk in page_chunks:
            chunks.append({
                "page": idx + 1,
                "text": chunk.strip()
            })
    return chunks

class RAGSystem:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        print(f"Loading and indexing PDF: {pdf_path}...")
        self.chunks = load_pdf_chunks(self.pdf_path)
        
        # Embed all chunks
        self.embeddings = []
        for idx, chunk in enumerate(self.chunks):
            res = genai.embed_content(
                model="models/gemini-embedding-001",
                content=chunk["text"],
                task_type="retrieval_document"
            )
            self.embeddings.append(res['embedding'])
        self.embeddings = np.array(self.embeddings)
        print(f"Indexed {len(self.chunks)} chunks successfully.")
        
    def retrieve(self, query, top_k=3):
        res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_emb = np.array(res['embedding'])
        
        # Compute cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        similarities = np.dot(self.embeddings, query_emb) / (norms * query_norm + 1e-10)
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        retrieved = []
        for idx in top_indices:
            retrieved.append(self.chunks[idx]["text"])
        return retrieved

    def answer_query(self, query):
        retrieved_contexts = self.retrieve(query)
        context_str = "\n---\n".join(retrieved_contexts)
        
        prompt = f"""
You are the AI Assistant for Mohammed Naseef's portfolio website. Your task is to answer the user's questions about Mohammed Naseef based on the context extracted from his resume (test11.pdf) below.

Context from test11.pdf:
{context_str}

User's Question:
{query}

Instructions:
1. Provide a professional, engaging, and accurate answer based on the provided context.
2. If the answer cannot be found in the context, use your general knowledge to answer nicely while acknowledging it's not explicitly in his resume, or ask the user to contact Mohammed Naseef directly.
3. Keep the response relatively concise (1-3 paragraphs) and format it nicely using markdown (e.g. bold text for key terms or bullet points where appropriate). Do not use HTML tags in your markdown output, just clean GFM.
"""
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text

current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_file_path = os.path.join(current_dir, "test11.pdf")
rag_system = RAGSystem(pdf_file_path)

# Initialize FastAPI App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty query message")
    try:
        ans = rag_system.answer_query(request.message)
        return {"response": ans}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static folder
app.mount("/", StaticFiles(directory=current_dir, html=True), name="static")
