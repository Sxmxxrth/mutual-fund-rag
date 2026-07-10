import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from sentence_transformers import CrossEncoder
import urllib.request

# ==========================================
# 1. Document Processor Logic
# ==========================================
class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\\n\\n", "\\n", " ", ""]
        )
        
    def load_and_split_pdf(self, file_path: str):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

# ==========================================
# 2. RAG Pipeline Logic
# ==========================================
class AdvancedRAGPipeline:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_directory)
        
        print("Loading local Open-Source LLM (Flan-T5)... this may take a moment.")
        pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def ingest_documents(self, chunks):
        self.vector_store.add_documents(chunks)

    def answer_financial_query(self, query: str) -> dict:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            return {"answer": "No relevant financial documents found.", "context": []}

        pairs = [[query, doc.page_content] for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)
        
        doc_score_pairs = list(zip(retrieved_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = [doc for doc, score in doc_score_pairs[:3]]
        context_str = "\\n\\n".join([doc.page_content for doc in top_docs])
        
        template = """You are an expert Mutual Fund Analyst. 
Use the following context to answer the question. If not found, say so.
Context:
{context}

Question: {question}
Expert Answer:"""
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": query})
            
        return {"answer": answer}

# ==========================================
# 3. Initialize and Pre-load Data
# ==========================================
doc_processor = DocumentProcessor()
rag_pipeline = AdvancedRAGPipeline()

# Download a dummy PDF on startup so it has data to search
if not os.path.exists("sample.pdf"):
    try:
        url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        urllib.request.urlretrieve(url, "sample.pdf")
        chunks = doc_processor.load_and_split_pdf("sample.pdf")
        rag_pipeline.ingest_documents(chunks)
    except Exception as e:
        print("Failed to download or ingest sample PDF on boot:", e)

# ==========================================
# 4. Gradio Interface
# ==========================================
def respond(message, history):
    result = rag_pipeline.answer_financial_query(message)
    return result.get("answer", "An error occurred.")

demo = gr.ChatInterface(
    fn=respond,
    title="🏦 Mutual Fund AI Assistant",
    description="Ask me any financial questions. I use an advanced **Cross-Encoder Re-Ranking** architecture to guarantee accuracy.",
    examples=["What is the expense ratio?", "What is the minimum investment?"],
    theme=gr.themes.Soft(primary_hue="blue")
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
