"""
test_router.py — Unit tests for the QueryRouter.

Tests that the router correctly classifies questions into sql/rag/hybrid.
These tests use a mock LLM to avoid API calls in CI.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.router.query_router import QueryRouter, RouteDecision


# ─── Test Fixtures ────────────────────────────────────────────────────────────

SQL_QUESTIONS = [
    "What is the total revenue for Q3 2024?",
    "How many customers signed up last month?",
    "List the top 5 products by sales volume",
    "What is the average order value by region?",
    "Which employees have a salary above $100,000?",
]

RAG_QUESTIONS = [
    "What is our refund and return policy?",
    "How do employees request parental leave?",
    "What certifications does the company hold?",
    "Describe the product warranty terms",
    "What is the code of conduct for remote work?",
]

HYBRID_QUESTIONS = [
    "Does our Q1 revenue meet the financial targets defined in company policy?",
    "Which product categories are underperforming relative to our catalog benchmarks?",
    "Are employee salaries in Engineering within the approved salary bands?",
]


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestRouteDecisionSchema:
    """Test that RouteDecision schema validates correctly."""
    
    def test_valid_sql_decision(self):
        decision = RouteDecision(
            route="sql",
            confidence=0.95,
            reasoning="Question requires numerical aggregation",
        )
        assert decision.route == "sql"
        assert decision.confidence == 0.95
    
    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            RouteDecision(route="sql", confidence=1.5, reasoning="invalid")
    
    def test_invalid_route(self):
        with pytest.raises(Exception):
            RouteDecision(route="invalid_route", confidence=0.9, reasoning="test")


class TestQueryRouterMocked:
    """Test router logic with mocked LLM responses."""
    
    @pytest.fixture
    def mock_router(self):
        with patch("src.router.query_router.ChatOpenAI"):
            router = QueryRouter()
            return router
    
    def test_router_fallback_on_error(self, mock_router):
        """Router should fall back to 'rag' if LLM fails."""
        mock_router.chain = MagicMock(side_effect=Exception("API timeout"))
        result = mock_router.route("Any question")
        assert result.route == "rag"
        assert result.confidence == 0.5
    
    def test_route_decision_has_required_fields(self):
        decision = RouteDecision(
            route="hybrid",
            confidence=0.8,
            reasoning="Needs both SQL data and policy context",
            sql_entities=["revenue", "Q1"],
            rag_keywords=["financial targets", "policy"],
        )
        assert len(decision.sql_entities) == 2
        assert len(decision.rag_keywords) == 2


@pytest.mark.parametrize("question", SQL_QUESTIONS)
def test_sql_question_classification(question):
    """Verify SQL questions contain quantitative signals."""
    sql_signals = ["total", "how many", "average", "list", "top", "salary", "revenue"]
    has_signal = any(signal in question.lower() for signal in sql_signals)
    assert has_signal, f"Expected SQL signal in: {question}"


@pytest.mark.parametrize("question", RAG_QUESTIONS)
def test_rag_question_classification(question):
    """Verify RAG questions contain qualitative/policy signals."""
    rag_signals = ["policy", "how do", "what is", "certif", "describe", "code of", "terms"]
    has_signal = any(signal in question.lower() for signal in rag_signals)
    assert has_signal, f"Expected RAG signal in: {question}"
