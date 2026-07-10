import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
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
    def __init__(self, persist_directory: str = "./data/chroma_db_openai"):
        self.persist_directory = persist_directory
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.demo_mode = False
        
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            self.demo_mode = True
            self.llm = None
            self.embeddings = None
            self.vector_store = None
            print("WARNING: No OpenAI API Key found. The pipeline cannot run without an API key.")
        else:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_directory)

    def ingest_documents(self, chunks):
        if not self.demo_mode:
            self.vector_store.add_documents(chunks)

    def answer_financial_query(self, query: str) -> dict:
        if self.demo_mode:
            return {"answer": "Error: You must set the OPENAI_API_KEY environment variable to run the API."}
            
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(query)
        
        if not retrieved_docs:
            return {"answer": "No relevant financial documents found.", "context": []}
        
        context_str = "\\n\\n".join([doc.page_content for doc in retrieved_docs])
        
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

if not rag_pipeline.demo_mode:
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
    title="🏦 Mutual Fund AI Assistant (100% API Powered)",
    description="This RAG architecture is powered completely by the OpenAI API (GPT-3.5-Turbo and Ada Embeddings) to minimize local computing footprint.",
    examples=["What is the expense ratio?", "What is the minimum investment?"],
    theme=gr.themes.Soft(primary_hue="blue")
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
