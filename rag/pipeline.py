"""
Mutual Fund RAG Pipeline - Core Implementation
Author: Samarth Sugandhi

Production-ready RAG pipeline for querying mutual fund documents.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "./data/chroma_db"
COLLECTION_NAME = "mutual_funds"


# ============================================================
# Document Ingestion
# ============================================================

class DocumentIngester:
    """Handles PDF ingestion, chunking, and embedding storage."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )

    def load_documents(self, directory: str) -> list:
        """Load all PDFs from a directory."""
        loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        documents = loader.load()
        print(f"📄 Loaded {len(documents)} pages from PDFs")
        return documents

    def chunk_documents(self, documents: list) -> list:
        """Split documents into chunks."""
        chunks = self.text_splitter.split_documents(documents)
        print(f"✂️  Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        return chunks

    def create_vector_store(
        self,
        chunks: list,
        persist_directory: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> Chroma:
        """Create and persist a ChromaDB vector store."""
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        print(f"💾 Vector store created at {persist_directory} with {len(chunks)} vectors")
        return vectorstore

    def ingest(self, input_dir: str) -> Chroma:
        """Full ingestion pipeline: load → chunk → embed → store."""
        documents = self.load_documents(input_dir)
        chunks = self.chunk_documents(documents)
        vectorstore = self.create_vector_store(chunks)
        return vectorstore


# ============================================================
# RAG Query Engine
# ============================================================

RAG_PROMPT_TEMPLATE = """You are an expert mutual fund advisor assistant. 
Use the following context from mutual fund documents to answer the question.

IMPORTANT RULES:
1. Only answer based on the provided context
2. If the answer is not in the context, say "I don't have enough information to answer this question"
3. Always cite which document/section the information comes from
4. Provide specific numbers, dates, and percentages when available
5. Be concise but thorough

Context:
{context}

Question: {question}

Answer:"""


class RAGQueryEngine:
    """Query engine for the mutual fund RAG pipeline."""

    def __init__(
        self,
        persist_directory: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = "gpt-4o-mini",
        top_k: int = 5,
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={"k": top_k, "fetch_k": top_k * 2},
        )
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.qa_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def query(self, question: str) -> dict:
        """
        Query the RAG pipeline.
        Returns answer and source documents.
        """
        result = self.qa_chain.invoke({"input": question})
        sources = [
            {
                "content": doc.page_content[:200],
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            }
            for doc in result.get("context", [])
        ]
        return {
            "answer": result["answer"],
            "sources": sources,
            "num_sources": len(sources),
        }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        input_dir = sys.argv[2] if len(sys.argv) > 2 else "./data/documents"
        ingester = DocumentIngester()
        ingester.ingest(input_dir)
        print("✅ Ingestion complete!")
    else:
        engine = RAGQueryEngine()
        question = " ".join(sys.argv[1:]) or "What are the top performing mutual funds?"
        print(f"\n❓ Question: {question}\n")
        result = engine.query(question)
        print(f"📝 Answer: {result['answer']}")
        print(f"\n📚 Sources ({result['num_sources']}):")
        for i, src in enumerate(result["sources"], 1):
            print(f"  {i}. {src['source']} (page {src['page']})")
