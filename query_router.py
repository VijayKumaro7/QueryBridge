"""
query_router.py — LLM-powered query classifier.

Classifies incoming natural language questions into one of three routes:
  - "sql"    → Structured data question, answered via Text-to-SQL
  - "rag"    → Unstructured/policy question, answered via vector retrieval
  - "hybrid" → Requires both SQL data AND document context

This is the brain of the system — it must be accurate because wrong routing
leads to wrong or missing answers.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─── Output Schema ────────────────────────────────────────────────────────────

class RouteDecision(BaseModel):
    """Structured routing decision returned by the LLM classifier."""
    
    route: Literal["sql", "rag", "hybrid"] = Field(
        description="The retrieval route to use for answering the question."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for this routing decision (0.0 to 1.0)."
    )
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen."
    )
    sql_entities: list[str] = Field(
        default_factory=list,
        description="Key entities/columns/tables relevant for SQL (if applicable)."
    )
    rag_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for semantic search in the vector store (if applicable)."
    )


# ─── Prompt ───────────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are an intelligent query routing agent for a hybrid SQL + RAG system.

Your job is to classify a user's question and decide HOW to retrieve the answer:

## Available Routes

### SQL Route
Use when the question requires:
- Numerical aggregations (SUM, AVG, COUNT, MIN, MAX)
- Filtering records (WHERE, HAVING)
- Sorting or ranking data
- Date-range queries
- Comparisons between structured data points
- Keywords: "how many", "total", "average", "list all", "top N", "between dates"

### RAG Route  
Use when the question requires:
- Qualitative information from documents/policies/manuals
- Explanations of processes or rules
- Company policies, procedures, guidelines
- Product descriptions or feature explanations
- Keywords: "what does the policy say", "explain", "describe", "according to"

### Hybrid Route
Use when the question requires BOTH:
- Comparing a metric from the database against a threshold defined in a document
- Enriching structured results with qualitative context
- Questions that blend "how much" with "what should it be"

## Database Schema Context
The SQL database contains:
- orders (order_id, customer_id, product_id, quantity, amount, order_date, region, status)
- customers (customer_id, name, email, segment, created_at)
- products (product_id, name, category, price, stock_quantity)
- employees (employee_id, name, department, salary, hire_date)

## Document Store Context
The vector store contains:
- Company HR policies and employee handbook
- Product catalog and feature descriptions
- Finance policies and approval thresholds
- Customer service guidelines

{format_instructions}
"""

ROUTER_HUMAN_PROMPT = """Classify this question and decide the retrieval route:

Question: {question}

Return your decision as structured JSON."""


# ─── Router Class ─────────────────────────────────────────────────────────────

class QueryRouter:
    """
    LLM-based query classifier that routes questions to the correct retrieval engine.
    
    Usage:
        router = QueryRouter()
        decision = router.route("What was total revenue in Q3?")
        print(decision.route)  # "sql"
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.router_model,
            temperature=0,       # Deterministic routing
            api_key=settings.openai_api_key,
        )
        self.parser = PydanticOutputParser(pydantic_object=RouteDecision)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", ROUTER_HUMAN_PROMPT),
        ]).partial(format_instructions=self.parser.get_format_instructions())
        
        self.chain = self.prompt | self.llm | self.parser
    
    def route(self, question: str) -> RouteDecision:
        """
        Classify a natural language question and return a routing decision.
        
        Args:
            question: The user's natural language question.
            
        Returns:
            RouteDecision with route, confidence, reasoning, and extracted keywords.
        """
        logger.info(f"Routing question: '{question[:80]}...'")
        
        try:
            decision = self.chain.invoke({"question": question})
            logger.info(
                f"Route: {decision.route} | Confidence: {decision.confidence:.2f} | "
                f"Reason: {decision.reasoning}"
            )
            return decision
        except Exception as e:
            logger.error(f"Router failed: {e}. Falling back to RAG.")
            # Safe fallback — RAG is less likely to cause SQL injection issues
            return RouteDecision(
                route="rag",
                confidence=0.5,
                reasoning=f"Fallback due to router error: {str(e)}",
            )
    
    async def aroute(self, question: str) -> RouteDecision:
        """Async version of route()."""
        logger.info(f"[async] Routing question: '{question[:80]}'")
        decision = await self.chain.ainvoke({"question": question})
        return decision
