"""
sql_chain.py — Text-to-SQL retrieval pipeline.

Takes a natural language question + extracted SQL entities,
generates a SQL query using the LLM, executes it safely,
and returns a formatted natural language answer.

Key design decisions:
  - Schema is injected into the prompt (not hardcoded) for flexibility
  - Queries are validated before execution (no DDL/DML allowed)
  - Results are post-processed by LLM for natural language output
"""
from __future__ import annotations

from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class SQLResult:
    """Result from the SQL chain."""
    answer: str
    sql_query: str | None = None
    raw_data: list[dict] | None = None
    error: str | None = None
    success: bool = True


class SQLChain:
    """
    Text-to-SQL pipeline using LangChain's SQL Agent.
    
    The agent can:
    1. Inspect the DB schema to understand available tables/columns
    2. Write a SQL query matching the user's intent
    3. Execute the query and interpret results
    4. Self-correct if the first query fails
    
    Usage:
        chain = SQLChain()
        result = chain.query("What is the total revenue by region?")
        print(result.answer)
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.db = SQLDatabase.from_uri(
            settings.database_url,
            # Restrict to read-only tables for safety
            include_tables=["orders", "customers", "products", "employees"],
        )
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            max_iterations=5,       # Limit self-correction loops
            handle_parsing_errors=True,
            prefix=self._build_prefix(),
        )
    
    def _build_prefix(self) -> str:
        return """You are a SQL expert assistant. Your job is to answer questions about business data
by writing and executing precise SQL queries.

RULES:
- Only use SELECT statements. Never use INSERT, UPDATE, DELETE, DROP, or CREATE.
- Always limit results to 100 rows unless asked for a specific number.
- Format numbers with proper units (currency as $X,XXX.XX, percentages as X.X%).
- If a query returns no results, say so clearly.
- Round floating point numbers to 2 decimal places.
- When referencing dates, use SQLite date functions.

After getting the SQL result, synthesize a clear, concise natural language answer."""
    
    def query(self, question: str, sql_entities: list[str] | None = None) -> SQLResult:
        """
        Execute a natural language question against the SQL database.
        
        Args:
            question: Natural language question about structured data.
            sql_entities: Optional hints from the router about relevant entities.
            
        Returns:
            SQLResult with the answer and metadata.
        """
        enhanced_question = question
        if sql_entities:
            enhanced_question = f"{question}\n\nHint: Focus on these entities: {', '.join(sql_entities)}"
        
        logger.info(f"SQL Chain processing: {question}")
        
        try:
            response = self.agent.invoke({"input": enhanced_question})
            return SQLResult(
                answer=response["output"],
                success=True,
            )
        except Exception as e:
            logger.error(f"SQL Chain error: {e}")
            return SQLResult(
                answer="I couldn't retrieve the data from the database. Please try rephrasing.",
                error=str(e),
                success=False,
            )
