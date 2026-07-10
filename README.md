# 🏦 Mutual Fund Advanced RAG Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** API designed specifically for analyzing complex financial documents, Mutual Fund Factsheets, and Prospectuses. 

This is not a standard tutorial RAG. It implements an advanced **Retrieve-and-ReRank** architecture to drastically reduce LLM hallucinations when dealing with exact financial figures.

## 🚀 Advanced Architecture Highlights

1. **Semantic Chunking:** Uses `RecursiveCharacterTextSplitter` heavily tuned (800 token chunk size, 150 token overlap) to ensure financial tables and paragraphs are not split midway.
2. **Dense Vector Retrieval:** Uses `BAAI/bge-small-en-v1.5` embeddings via **ChromaDB** for hyper-fast Initial Retrieval of the top 10 relevant context chunks.
3. **Cross-Encoder Re-Ranking:** **(The Key Differentiator)** Standard Cosine Similarity is often flawed for specific text comparisons. This pipeline passes the Top 10 retrieved chunks through a Deep Learning `Cross-Encoder` (`ms-marco-MiniLM-L-6-v2`) which mathematically scores the exact relevance between the user's query and the chunk, re-ranking them to ensure the LLM only sees the absolute most relevant data.
4. **LLM Synthesis:** Synthesizes the finalized context securely using **LangChain Expression Language (LCEL)** and OpenAI's models with a strict financial prompt preventing hallucination.
5. **Production Backend:** Fully decoupled API built with **FastAPI** for instant frontend integration.

## 🛠 Tech Stack
- **API Framework:** FastAPI, Uvicorn, Pydantic
- **RAG Orchestration:** LangChain (LCEL)
- **Vector Database:** ChromaDB
- **Embeddings & Re-Ranking:** HuggingFace `sentence-transformers`, `Cross-Encoder`
- **LLM:** OpenAI (`gpt-3.5-turbo`)

## 💻 How to Run Locally

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the environment variables template and add your OpenAI API Key:
   ```bash
   cp .env.example .env
   # Edit .env and insert your OPENAI_API_KEY
   ```
3. Boot the FastAPI Server:
   ```bash
   uvicorn main:app --reload
   ```
4. Access the interactive Swagger UI at: `http://localhost:8000/docs`

## 📊 API Endpoints
* `POST /api/v1/ingest`: Upload a Mutual Fund PDF to instantly vectorize and store it in ChromaDB.
* `POST /api/v1/query`: Ask financial questions (e.g. *"What is the expense ratio?"*) and receive precise, cross-encoder validated answers.
