# 📄 Mutual Fund RAG Assistant

> A production-ready Retrieval-Augmented Generation system for querying mutual fund documents using LangChain, ChromaDB, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Problem Statement

Mutual fund documents (SIDs, factsheets, annual reports) are dense, lengthy PDFs that are difficult to navigate. Investors and advisors need quick, accurate answers from these documents.

**Solution**: A RAG pipeline that ingests mutual fund PDFs, creates semantic embeddings, and enables natural language Q&A with source citations.

## 🏗️ Architecture

```
PDF Documents → [Ingestion Pipeline] → [Chunking] → [Embeddings] → ChromaDB
                                                                        ↓
User Question → [Query Engine] → [Semantic Search] → [LLM + Context] → Answer
```

### Pipeline Components:
1. **Document Ingestion**: PDF parsing with PyPDF2/pdfplumber
2. **Text Chunking**: RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
3. **Embedding Generation**: sentence-transformers/all-MiniLM-L6-v2
4. **Vector Storage**: ChromaDB (persistent storage)
5. **Retrieval**: Hybrid search (semantic + keyword BM25)
6. **Generation**: LLM with retrieved context + source citations

## 📊 Benchmark Results (RAGAS)

| Metric | Score |
|--------|-------|
| Context Precision | **0.81** |
| Answer Relevancy | **0.78** |
| Faithfulness | **0.85** |
| Context Recall | **0.76** |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers |
| LLM | OpenAI GPT-4 / Local (Ollama) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Evaluation | RAGAS |

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/Sxmxxrth/mutual-fund-rag.git
cd mutual-fund-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest Documents
```bash
# Place PDFs in the data/documents/ directory
python ingest.py --input data/documents/ --collection mutual_funds
```

### 3. Run the App
```bash
# Streamlit UI
streamlit run app.py

# FastAPI backend
uvicorn api:app --reload --port 8000
```

### 4. Query
```bash
# API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the expense ratio of HDFC Balanced Fund?"}'
```

## 📁 Project Structure

```
mutual-fund-rag/
├── app.py               # Streamlit frontend
├── api.py               # FastAPI backend
├── ingest.py            # Document ingestion script
├── rag/
│   ├── __init__.py
│   ├── chunker.py       # Text chunking strategies
│   ├── embedder.py      # Embedding generation
│   ├── retriever.py     # Hybrid retrieval (semantic + BM25)
│   ├── generator.py     # LLM response generation
│   └── pipeline.py      # End-to-end RAG pipeline
├── evaluation/
│   ├── evaluate.py      # RAGAS evaluation script
│   └── test_questions.json
├── data/
│   ├── documents/       # Input PDFs
│   └── chroma_db/       # Vector store
├── config/
│   └── settings.py
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🧪 Testing

```bash
# Unit tests
pytest tests/ -v

# RAG evaluation
python evaluation/evaluate.py
```

## 🐳 Docker

```bash
docker build -t mutual-fund-rag .
docker run -p 8501:8501 --env-file .env mutual-fund-rag
```

## 💡 Key Design Decisions

1. **Hybrid Retrieval**: Combining semantic search with BM25 keyword matching improves recall by ~15% over pure semantic search
2. **Chunk Size**: 512 tokens with 50-token overlap balances context completeness with retrieval precision
3. **Prompt Engineering**: Custom prompt template with explicit instructions to cite sources and handle "I don't know" cases
4. **Evaluation**: RAGAS framework for systematic quality measurement

## 📝 License

MIT License

## 👤 Author

**Samarth Sugandhi**
- GitHub: [@Sxmxxrth](https://github.com/Sxmxxrth)
- LinkedIn: [samarthz](https://linkedin.com/in/samarthz)
- Email: samarthz.icloud@gmail.com
