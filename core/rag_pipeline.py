import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from sentence_transformers import CrossEncoder

load_dotenv()

class AdvancedRAGPipeline:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        
        # 1. Base Embedding Model (Dense Vector Search)
        # Using a highly-performant open-source embedding model tailored for semantic search
        print("Initializing Dense Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # 2. Advanced Concept: Cross-Encoder Re-Ranker
        # After retrieving the top K documents, a cross-encoder scores the exact 
        # relevance between the Query and the Document to re-rank them for maximum accuracy.
        print("Initializing Cross-Encoder Re-Ranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # 3. Vector Store
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # 4. LLM for Generation
        # Requires OPENAI_API_KEY in .env
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print("WARNING: OPENAI_API_KEY not set in .env. LLM generation will fail.")
            self.llm = None
        else:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def ingest_documents(self, chunks):
        """Stores document chunks into the Vector Database."""
        print(f"Ingesting {len(chunks)} chunks into ChromaDB...")
        self.vector_store.add_documents(chunks)
        print("Ingestion complete!")

    def answer_financial_query(self, query: str) -> dict:
        """
        Executes the Advanced RAG Pipeline:
        Retrieval -> Cross-Encoder Re-Ranking -> LLM Synthesis
        """
        if not self.llm:
            return {"error": "LLM not initialized. Please set OPENAI_API_KEY in .env"}

        # Step 1: Initial Dense Retrieval (Fetch top 10 candidates)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            return {"answer": "No relevant financial documents found in the database to answer this query.", "context": []}

        # Step 2: Cross-Encoder Re-Ranking
        # We pair the query with each document and ask the cross-encoder to score the exact relevance.
        pairs = [[query, doc.page_content] for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)
        
        # Sort documents by their re-ranker score in descending order
        doc_score_pairs = list(zip(retrieved_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the top 3 most highly relevant documents to fit inside the LLM context window securely
        top_docs = [doc for doc, score in doc_score_pairs[:3]]
        context_str = "\\n\\n".join([doc.page_content for doc in top_docs])
        
        # Step 3: LLM Synthesis with strict Financial Prompt
        template = """You are an expert Mutual Fund Analyst assistant. 
Use the following pieces of retrieved financial context to answer the user's question. 
If the answer is not contained within the provided context, explicitly state "I cannot answer this based on the provided documents." Do not hallucinate financial numbers.

Context:
{context}

Question: {question}

Expert Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # LangChain Expression Language (LCEL) Chain
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context_str,
            "question": query
        })
        
        return {
            "answer": answer,
            "top_retrieved_sources": [doc.metadata.get('source', 'Unknown') for doc in top_docs]
        }
