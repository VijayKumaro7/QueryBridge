"""
ingest_docs.py — Load documents into ChromaDB vector store.

Supports: .txt, .md, .pdf files
Chunks documents and generates embeddings for semantic search.

Run once before starting the application:
    python scripts/ingest_docs.py
"""
import os
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader, UnstructuredMarkdownLoader, PyPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path("data/sample_docs")
CHROMA_DIR = Path("data/chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "company_docs")

LOADERS = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PyPDFLoader,
}

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_documents():
    docs = []
    for file_path in DOCS_DIR.rglob("*"):
        if file_path.suffix in LOADERS:
            try:
                loader_cls = LOADERS[file_path.suffix]
                loader = loader_cls(str(file_path))
                file_docs = loader.load()
                # Tag with source metadata
                for doc in file_docs:
                    doc.metadata["source"] = file_path.name
                    doc.metadata["file_path"] = str(file_path)
                docs.extend(file_docs)
                print(f"  ✅ Loaded: {file_path.name} ({len(file_docs)} docs)")
            except Exception as e:
                print(f"  ❌ Failed to load {file_path.name}: {e}")
    return docs


def ingest():
    print(f"Loading documents from: {DOCS_DIR.absolute()}")
    docs = load_documents()
    
    if not docs:
        print("⚠️  No documents found. Add files to data/sample_docs/")
        return
    
    print(f"\nSplitting {len(docs)} documents into chunks...")
    chunks = SPLITTER.split_documents(docs)
    print(f"  Created {len(chunks)} chunks")
    
    print(f"\nGenerating embeddings and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    
    count = vectorstore._collection.count()
    print(f"\n🎉 Done! {count} chunks stored in ChromaDB at {CHROMA_DIR.absolute()}")


if __name__ == "__main__":
    ingest()
