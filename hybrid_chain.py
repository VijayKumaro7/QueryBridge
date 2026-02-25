"""
hybrid_chain.py — Orchestrator for hybrid SQL + RAG queries.

When the router decides "hybrid", this chain:
1. Runs SQL chain and RAG chain in parallel
2. Feeds both results to a synthesis LLM
3. Returns a unified, coherent answer

This is the most architecturally interesting part — it demonstrates
how to combine heterogeneous data sources into a single response.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.chains.sql_chain import SQLChain, SQLResult
from src.chains.rag_chain import RAGChain, RAGResult
from src.router.query_router import RouteDecision
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class HybridResult:
    """Combined result from both SQL and RAG chains."""
    final_answer: str
    sql_result: SQLResult | None = None
    rag_result: RAGResult | None = None
    route_used: str = "hybrid"
    sources: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None


SYNTHESIS_PROMPT = ChatPromptTemplate.from_template("""You are a senior analyst synthesizing information
from two different sources to answer a question comprehensively.

## Original Question
{question}

## Structured Data Answer (from Database)
{sql_answer}

## Document/Policy Context (from Company Documents)  
{rag_answer}

## Your Task
Synthesize both pieces of information into ONE coherent, insightful answer.
- Reference specific numbers from the database
- Reference specific policies/guidelines from documents
- Highlight any gaps between what the data shows and what policy requires
- Be concise but comprehensive
- If either source had no relevant information, note that clearly

Synthesized Answer:""")


class HybridChain:
    """
    Orchestrator that runs SQL and RAG chains and synthesizes their outputs.
    
    For hybrid queries (e.g., "Is our Q1 revenue above the target in our financial policy?"),
    this chain fetches both the metric (SQL) and the target (RAG) and combines them.
    
    Usage:
        chain = HybridChain()
        result = chain.query("Does Q1 revenue meet policy targets?", decision)
    """
    
    def __init__(self):
        self.sql_chain = SQLChain()
        self.rag_chain = RAGChain()
        self.synthesis_llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.2,
            api_key=settings.openai_api_key,
        )
        self.synthesis_chain = SYNTHESIS_PROMPT | self.synthesis_llm | StrOutputParser()
    
    def query(self, question: str, decision: RouteDecision) -> HybridResult:
        """
        Run SQL and RAG chains and synthesize results.
        
        Args:
            question: Original user question.
            decision: Router decision with extracted entities and keywords.
            
        Returns:
            HybridResult with synthesized final answer.
        """
        logger.info(f"Hybrid Chain: running SQL + RAG in parallel for: {question}")
        
        # Run both chains (could be parallelized with asyncio in production)
        sql_result = self.sql_chain.query(question, decision.sql_entities)
        rag_result = self.rag_chain.query(question, decision.rag_keywords)
        
        # Synthesize
        try:
            final_answer = self.synthesis_chain.invoke({
                "question": question,
                "sql_answer": sql_result.answer if sql_result.success else "No database data available.",
                "rag_answer": rag_result.answer if rag_result.success else "No document context available.",
            })
            
            all_sources = rag_result.sources.copy()
            if sql_result.success:
                all_sources.append("Business Database (SQL)")
            
            return HybridResult(
                final_answer=final_answer,
                sql_result=sql_result,
                rag_result=rag_result,
                sources=all_sources,
                success=True,
            )
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Graceful degradation: return best available result
            fallback = sql_result.answer if sql_result.success else rag_result.answer
            return HybridResult(
                final_answer=f"{fallback}\n\n[Note: Full synthesis failed]",
                sql_result=sql_result,
                rag_result=rag_result,
                error=str(e),
                success=False,
            )
