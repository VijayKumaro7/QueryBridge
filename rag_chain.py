"""
rag_chain.py — RAG (Retrieval-Augmented Generation) pipeline.

Takes a natural language question, retrieves semantically relevant
document chunks from ChromaDB, and generates a grounded answer.

Key design decisions:
  - MMR (Maximal Marginal Relevance) for diverse chunk retrieval
  - Source citation included in every answer
  - Configurable chunk count (k) for precision vs. recall tradeoff
  - Confidence score based on retrieval similarity
"""
from __future__ import annotations

from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RAGResult:
    """Result from the RAG chain."""
    answer: str
    sources: list[str] = field(default_factory=list)
    retrieved_chunks: int = 0
    avg_similarity: float = 0.0
    success: bool = True
    error: str | None = None


RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions
based on company documents and policies.

Use ONLY the context below to answer the question. If the context doesn't contain
enough information to answer confidently, say "I don't have enough information in 
the documents to answer this question."

Always cite which document your answer comes from.

Context:
{context}

Question: {question}

Answer (with source citations):""")


def format_docs(docs) -> str:
    """Format retrieved documents into a readable context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Document")
        formatted.append(f"[{i}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


class RAGChain:
    """
    Retrieval-Augmented Generation pipeline using ChromaDB + OpenAI.
    
    The chain:
    1. Embeds the user question
    2. Retrieves top-k relevant chunks via MMR (Maximal Marginal Relevance)
    3. Injects chunks as context into a grounded answering prompt
    4. Returns answer with source citations
    
    Usage:
        chain = RAGChain()
        result = chain.query("What is the refund policy?")
        print(result.answer)
        print(result.sources)
    """
    
    def __init__(self, k: int = 4):
        """
        Args:
            k: Number of document chunks to retrieve per query.
        """
        self.k = k
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1,
            api_key=settings.openai_api_key,
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        self.vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        # MMR retrieval: balances relevance AND diversity of chunks
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.k * 3, "lambda_mult": 0.7},
        )
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str, rag_keywords: list[str] | None = None) -> RAGResult:
        """
        Answer a question using document retrieval.
        
        Args:
            question: Natural language question about documents/policies.
            rag_keywords: Optional keywords from router to enhance retrieval.
            
        Returns:
            RAGResult with answer and source citations.
        """
        search_query = question
        if rag_keywords:
            # Enrich query with extracted keywords for better retrieval
            search_query = f"{question} {' '.join(rag_keywords)}"
        
        logger.info(f"RAG Chain processing: {question}")
        
        try:
            # Retrieve documents for metadata
            docs = self.retriever.invoke(search_query)
            sources = list(set(
                doc.metadata.get("source", "Unknown") for doc in docs
            ))
            
            # Generate answer
            answer = self.chain.invoke(search_query)
            
            return RAGResult(
                answer=answer,
                sources=sources,
                retrieved_chunks=len(docs),
                success=True,
            )
        except Exception as e:
            logger.error(f"RAG Chain error: {e}")
            return RAGResult(
                answer="I couldn't retrieve information from the documents. Please try again.",
                error=str(e),
                success=False,
            )
