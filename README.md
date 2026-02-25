# 🌉 QueryBridge

> **Bridge the gap between structured data and unstructured knowledge** — QueryBridge is a production-grade intelligent query router that decides in real-time whether to answer from a SQL database, a vector document store, or both — and synthesizes a unified response.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?logo=chainlink)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-red?logo=fastapi)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🧠 What Makes QueryBridge Different

Most LLM projects either do **Text-to-SQL** OR **RAG**. QueryBridge does **both** — intelligently.

When a user asks a question, the **Router Agent** classifies the intent and routes it to the right retrieval engine:

| Question Type | Route | Engine |
|---|---|---|
| *"What is the total revenue in Q3?"* | → SQL | SQLite + LangChain SQLAgent |
| *"What does our refund policy say?"* | → RAG | ChromaDB + Semantic Search |
| *"Compare Q3 revenue with the policy threshold"* | → Hybrid | Both + LLM Synthesis |

This architectural decision-making is what makes QueryBridge portfolio-worthy.

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Router Agent (LLM)          │  ← Classifies intent: SQL / RAG / Hybrid
└──────────┬──────────────┬───────────┘
           │              │
     ┌─────▼─────┐  ┌────▼──────┐
     │ SQL Chain │  │ RAG Chain │
     │           │  │           │
     │ Text-to-  │  │ Embedding │
     │   SQL     │  │  + Search │
     │  Agent    │  │           │
     └─────┬─────┘  └────┬──────┘
           │              │
     ┌─────▼─────┐  ┌────▼──────┐
     │ SQLite DB │  │ ChromaDB  │
     │(Structured│  │(Unstructur│
     │   Data)   │  │   ed)     │
     └─────┬─────┘  └────┬──────┘
           │              │
           └──────┬───────┘
                  ▼
         ┌────────────────┐
         │ Response Synth │  ← Merges answers when both are used
         │    (LLM)       │
         └────────────────┘
                  │
                  ▼
             Final Answer
```

---

## 📁 Project Structure

```
QueryBridge/
│
├── src/
│   ├── router/
│   │   ├── __init__.py
│   │   ├── query_router.py         # LLM-based intent classifier
│   │   └── router_prompts.py       # Prompt templates for routing
│   │
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── sql_chain.py            # Text-to-SQL LangChain pipeline
│   │   ├── rag_chain.py            # RAG retrieval + generation pipeline
│   │   └── hybrid_chain.py         # Orchestrator for combined queries
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── db_manager.py           # SQLite connection & schema management
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── chroma_manager.py       # ChromaDB CRUD operations
│   │   └── document_loader.py      # PDF/TXT/MD ingestion pipeline
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   └── synthesis_agent.py      # Final answer synthesizer
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                 # FastAPI application entry point
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # Structured logging setup
│       └── config.py               # Environment & settings management
│
├── frontend/
│   ├── index.html                  # Zero-dependency chat UI (Demo Mode built-in)
│   └── README.md
│
├── data/
│   ├── sample_db/
│   │   └── business.db             # Pre-seeded SQLite database
│   └── sample_docs/
│       ├── company_policy.txt      # Refund, approval & expense policies
│       ├── employee_handbook.md    # Leave policies & salary bands
│       ├── finance_policy.md       # Revenue targets & financial rules
│       └── product_catalog.md      # Product features & pricing
│
├── tests/
│   ├── __init__.py
│   ├── test_router.py              # Unit tests for query classification
│   ├── test_sql_chain.py           # SQL generation accuracy tests
│   ├── test_rag_chain.py           # Retrieval relevance tests
│   └── test_hybrid.py              # End-to-end integration tests
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Explore sample database
│   ├── 02_router_evaluation.ipynb  # Test routing accuracy on benchmarks
│   └── 03_full_demo.ipynb          # Interactive end-to-end walkthrough
│
├── docs/
│   ├── architecture.md             # Deep-dive into system design
│   ├── routing_logic.md            # How the router makes decisions
│   └── api_reference.md            # REST API documentation
│
├── scripts/
│   ├── setup_db.py                 # Initialize and seed the database
│   ├── ingest_docs.py              # Load documents into ChromaDB
│   └── evaluate.py                 # Run evaluation benchmarks
│
├── .env.example                    # Environment variable template
├── .gitignore
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | OpenAI GPT-4o / Ollama | Query routing + generation |
| **Orchestration** | LangChain 0.2+ | Chain & agent management |
| **SQL Engine** | SQLite + SQLAlchemy | Structured data storage |
| **Vector Store** | ChromaDB | Semantic document retrieval |
| **Embeddings** | OpenAI `text-embedding-3-small` | Document + query vectorization |
| **API** | FastAPI | REST interface |
| **Frontend** | Vanilla HTML/CSS/JS | Zero-dependency chat UI |
| **Testing** | Pytest | Unit & integration tests |
| **Config** | Pydantic Settings | Type-safe environment management |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/VijayKumaro7/QueryBridge.git
cd QueryBridge

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Initialize Database & Vector Store

```bash
python scripts/setup_db.py        # Seeds SQLite with 1,200 orders, 30 customers, 15 employees
python scripts/ingest_docs.py     # Loads 4 sample policy docs into ChromaDB
```

### 4. Run the API

```bash
uvicorn src.api.main:app --reload
# Swagger docs: http://localhost:8000/docs
```

### 5. Open the UI

```bash
open frontend/index.html
# Or: python -m http.server 3000 --directory frontend
```

### 6. Ask a Question via curl

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total sales revenue last quarter?"}'
```

---

## 💬 Example Queries & Routing

```python
# SQL Route → Generates and executes SQL against SQLite
"What is the average order value by region?"
"List the top 5 products by revenue"
"How many customers signed up last month?"
"Show Engineering department salaries"

# RAG Route → Semantic search over policy documents
"What is our return and refund policy?"
"How do employees request parental leave?"
"What are the approved salary bands for Engineering?"
"Tell me about the Analytics Pro product"

# Hybrid Route → Both engines + LLM synthesis
"Does our Q3 revenue meet the targets defined in financial policy?"
"Are Engineering salaries within the approved compensation bands?"
"Which products are underperforming compared to catalog benchmarks?"
```

---

## 🔬 How the Router Works

QueryBridge's `QueryRouter` uses a structured LLM prompt to classify queries into one of three categories:

```python
class RouteDecision(BaseModel):
    route: Literal["sql", "rag", "hybrid"]
    confidence: float          # 0.0 - 1.0
    reasoning: str             # Why this route was chosen
    sql_entities: list[str]    # Extracted entities for SQL
    rag_keywords: list[str]    # Keywords for vector search
```

**Routing signals the LLM is guided by:**
- **SQL signals**: numbers, aggregations, comparisons, dates, "how many", "total", "average", "top N"
- **RAG signals**: "policy", "document", "explain", "what does it say about", qualitative questions
- **Hybrid signals**: questions mixing metrics with context or policy thresholds

---

## 🖥️ Frontend UI

QueryBridge ships with a zero-dependency chat interface — no npm, no build step required.

```bash
# Open directly in browser
open frontend/index.html

# Or serve with Python
python -m http.server 3000 --directory frontend
```

**UI Features:**
- ⚡ **Demo Mode** — works instantly with built-in mock data, no backend needed
- 🔀 **Live pipeline visualization** — see the exact SQL → RAG → Hybrid path per query
- 📊 **Confidence meter** — animated bar showing router certainty
- 🎛️ **Force route toggle** — override auto routing for demos and debugging
- 🗂️ **Live data preview** — browse all 4 database tables in the sidebar
- 📜 **Query history** — sidebar tracks all past queries

---

## 📊 Evaluation Metrics

QueryBridge includes an evaluation suite (`scripts/evaluate.py`) measuring:

| Metric | Description |
|---|---|
| **Routing Accuracy** | % of queries correctly classified (SQL/RAG/Hybrid) |
| **SQL Execution Rate** | % of generated SQL that runs without error |
| **SQL Answer Accuracy** | % of SQL results matching ground truth |
| **RAG Relevance Score** | Mean cosine similarity of retrieved chunks |
| **End-to-End Latency** | P50/P95 response time per route |

---

## 🐳 Docker Setup

```bash
docker-compose up --build
# API available at http://localhost:8000
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
pytest tests/test_router.py -v --tb=short   # Router only
```

---

## 🗺️ Roadmap

- [x] Query Router with LLM-based classification
- [x] Text-to-SQL chain with schema-aware prompting
- [x] RAG pipeline with ChromaDB
- [x] Hybrid synthesis agent
- [x] FastAPI REST interface
- [x] Zero-dependency frontend with Demo Mode
- [x] Sample dataset (1,200 orders, 4 policy documents)
- [ ] Streaming responses via WebSocket
- [ ] Query result caching (Redis)
- [ ] Multi-tenant vector store namespacing
- [ ] LangSmith tracing integration
- [ ] Streamlit demo UI

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/streaming-responses`)
3. Commit with clear messages (`git commit -m "feat: add streaming support to RAG chain"`)
4. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Vijay** — Data Analyst & ML Engineer  
[GitHub](https://github.com/VijayKumaro7) · [LinkedIn](https://linkedin.com/in/yourprofile)

> *QueryBridge — bridging structured databases and unstructured knowledge with intelligent routing.*
