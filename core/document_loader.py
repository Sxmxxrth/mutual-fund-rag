import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # We use a RecursiveCharacterTextSplitter which is optimal for financial 
        # documents as it tries to keep paragraphs and sentences together.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\\n\\n", "\\n", " ", ""]
        )
        
    def load_and_split_pdf(self, file_path: str):
        """
        Loads a PDF (e.g. Mutual Fund Factsheet) and splits it into semantic chunks.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        print(f"Loading document: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages. Splitting into semantic chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Successfully generated {len(chunks)} chunks.")
        
        return chunks
