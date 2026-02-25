"""
main.py — FastAPI application entry point.

Exposes a REST API for the SQL + RAG Hybrid system.
The main endpoint accepts a question and returns an answer
with metadata about which route was used.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from src.router.query_router import QueryRouter, RouteDecision
from src.chains.sql_chain import SQLChain
from src.chains.rag_chain import RAGChain
from src.chains.hybrid_chain import HybridChain
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="SQL + RAG Hybrid Intelligence API",
    description="""
    A hybrid query system that intelligently routes natural language questions
    to either a SQL database, a vector document store, or both.
    
    **Routes:**
    - `/query` — Main endpoint: ask any question
    - `/health` — Health check
    - `/route` — Preview routing decision without executing
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialization (avoids cold-start on import)
_router: QueryRouter | None = None
_sql_chain: SQLChain | None = None
_rag_chain: RAGChain | None = None
_hybrid_chain: HybridChain | None = None


def get_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router


def get_chains():
    global _sql_chain, _rag_chain, _hybrid_chain
    if _sql_chain is None:
        _sql_chain = SQLChain()
        _rag_chain = RAGChain()
        _hybrid_chain = HybridChain()
    return _sql_chain, _rag_chain, _hybrid_chain


# ─── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000, example="What was total revenue last quarter?")
    force_route: Literal["sql", "rag", "hybrid"] | None = Field(
        None, description="Override automatic routing (for testing)"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    route_used: Literal["sql", "rag", "hybrid"]
    confidence: float
    routing_reason: str
    sources: list[str]
    success: bool


class RoutePreviewResponse(BaseModel):
    question: str
    predicted_route: str
    confidence: float
    reasoning: str
    sql_entities: list[str]
    rag_keywords: list[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "sql-rag-hybrid"}


@app.post("/route", response_model=RoutePreviewResponse)
def preview_route(request: QueryRequest):
    """
    Preview how the system would route a question without executing it.
    Useful for debugging and understanding router behavior.
    """
    router = get_router()
    decision: RouteDecision = router.route(request.question)
    return RoutePreviewResponse(
        question=request.question,
        predicted_route=decision.route,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        sql_entities=decision.sql_entities,
        rag_keywords=decision.rag_keywords,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main endpoint: routes the question and returns an answer.
    
    The system automatically decides whether to query the SQL database,
    search documents, or combine both approaches.
    """
    router = get_router()
    sql_chain, rag_chain, hybrid_chain = get_chains()
    
    # Step 1: Route the question
    decision: RouteDecision = router.route(request.question)
    effective_route = request.force_route or decision.route
    
    logger.info(f"Processing query via route: {effective_route}")
    
    # Step 2: Execute the right chain
    sources = []
    
    if effective_route == "sql":
        result = sql_chain.query(request.question, decision.sql_entities)
        answer = result.answer
        sources = ["Business Database (SQL)"]
        success = result.success
        
    elif effective_route == "rag":
        result = rag_chain.query(request.question, decision.rag_keywords)
        answer = result.answer
        sources = result.sources
        success = result.success
        
    elif effective_route == "hybrid":
        result = hybrid_chain.query(request.question, decision)
        answer = result.final_answer
        sources = result.sources
        success = result.success
    else:
        raise HTTPException(status_code=400, detail=f"Unknown route: {effective_route}")
    
    return QueryResponse(
        question=request.question,
        answer=answer,
        route_used=effective_route,
        confidence=decision.confidence,
        routing_reason=decision.reasoning,
        sources=sources,
        success=success,
    )
