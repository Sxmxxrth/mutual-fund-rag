# 🏦 Mutual Fund RAG Assistant

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end Retrieval-Augmented Generation (RAG) system that converts complex mutual fund PDFs into a highly accurate, searchable vector database for natural language Q&A.

## 🌟 Key Engineering Features
- **Advanced Retrieval**: Utilizes ChromaDB and FAISS with Maximal Marginal Relevance (MMR) for diverse and highly accurate context fetching.
- **Modern LangChain LCEL**: Fully updated to use the latest LangChain Expression Language (`create_retrieval_chain`) for robust orchestration.
- **RAGAS Evaluated**: Achieved **0.81 Context Precision** using the RAGAS evaluation framework.
- **Containerized Stack**: Fully dockerized application ensuring a reproducible environment.

## 🏗️ System Architecture
`User UI (Streamlit) -> LCEL Pipeline -> Embeddings (SentenceTransformers) -> Vector Store (ChromaDB) -> LLM (GPT-4o-mini)`

## 🚀 Quick Start (Using Docker)

The easiest way to run this project is via Docker. Ensure you have Docker and docker-compose installed.

```bash
# 1. Clone the repository
git clone https://github.com/Sxmxxrth/mutual-fund-rag.git
cd mutual-fund-rag

# 2. Add your API Key
export OPENAI_API_KEY="your-api-key-here"

# 3. Start the application
make docker-up
```

Access the UI at `http://localhost:8501`.

## 🧪 Development (Local Setup)

If you want to run it without Docker:
```bash
# Install dependencies
make install

# Optional: Ingest new PDFs placed in ./data/documents
make ingest

# Run the UI
make run
```
