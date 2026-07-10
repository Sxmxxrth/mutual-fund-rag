from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil

from core.document_loader import DocumentProcessor
from core.rag_pipeline import AdvancedRAGPipeline

app = FastAPI(
    title="Mutual Fund Advanced RAG Assistant",
    description="A production-grade Retrieval-Augmented Generation API featuring Cross-Encoder Re-Ranking.",
    version="1.0.0"
)

# Initialize singletons for our ML pipeline
doc_processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)
rag_pipeline = AdvancedRAGPipeline()

# Ensure temp directory exists for uploads
os.makedirs("temp_uploads", exist_ok=True)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "Active", "architecture": "Advanced RAG with Re-Ranking"}

@app.post("/api/v1/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a Mutual Fund Factsheet (PDF).
    It will automatically chunk the text and persist the dense embeddings in ChromaDB.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
        
    temp_path = os.path.join("temp_uploads", file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Load and split
        chunks = doc_processor.load_and_split_pdf(temp_path)
        
        # Ingest into VectorDB
        rag_pipeline.ingest_documents(chunks)
        
        return {
            "message": f"Successfully ingested {file.filename}",
            "chunks_processed": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/v1/query")
def query_financial_assistant(request: QueryRequest):
    """
    Endpoint to ask complex financial questions about the ingested Mutual Funds.
    Utilizes Dense Retrieval followed by Cross-Encoder Re-Ranking for maximum accuracy.
    """
    response = rag_pipeline.answer_financial_query(request.query)
    
    if "error" in response:
        raise HTTPException(status_code=500, detail=response["error"])
        
    return response
