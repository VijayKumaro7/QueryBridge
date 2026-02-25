# Architecture Deep Dive

## The Core Problem

Most enterprise data lives in two places:
1. **Structured databases** — transactional data (orders, customers, sales)
2. **Unstructured documents** — policies, manuals, reports, catalogs

Traditional chatbots handle one or the other. This system handles **both** by intelligently routing.

---

## Routing Decision Tree

```
User Query
    │
    ▼
Does the question require numerical/tabular data?
    ├── YES → Does it also need document context?
    │               ├── YES → HYBRID
    │               └── NO  → SQL
    └── NO  → RAG
```

### SQL Signals (high-confidence indicators)
- Aggregation verbs: "total", "average", "count", "sum", "max", "min"
- Ranking: "top N", "bottom N", "most", "least"
- Filtering: "where", "in region", "between dates"
- Comparative: "more than", "less than", "above", "below"

### RAG Signals
- Policy questions: "what does the policy say", "according to"
- Qualitative: "explain", "describe", "how does it work"
- Document references: "handbook", "manual", "guideline", "terms"

### Hybrid Signals
- Mixed intent: metric + context ("Does our performance meet our policy target?")
- Enrichment: "what does the data show compared to our guidelines?"

---

## Data Flow Detail

### SQL Path
```
Question → Router (SQL) → Schema Injection → LLM generates SQL
→ SQLite executor → Raw results → LLM interprets → Natural language answer
```

### RAG Path
```
Question → Router (RAG) → Embed question → ChromaDB MMR search
→ Top-k chunks retrieved → Context window injection → LLM generates answer
→ Source citations appended
```

### Hybrid Path
```
Question → Router (Hybrid) → [SQL Path + RAG Path] in parallel
→ Both answers → Synthesis LLM → Unified answer with citations
```

---

## Why MMR for Retrieval?

Standard cosine similarity retrieval can return 4 nearly-identical chunks from the same paragraph. **Maximal Marginal Relevance** balances:
- **Relevance** to the query
- **Diversity** among retrieved chunks

This gives more comprehensive answers by covering different aspects of the topic.

---

## Schema-Aware SQL Generation

The SQL agent is given the full database schema at every turn. This means:
- It knows column names and types without hardcoding
- It can JOIN across tables intelligently
- Adding new tables only requires updating the `include_tables` list

---

## Failure Modes & Mitigations

| Failure | Mitigation |
|---|---|
| Router misclassifies | `force_route` parameter for manual override |
| SQL query errors | Agent self-corrects up to 5 times |
| No relevant docs found | Explicit "I don't know" response |
| LLM API timeout | Fallback to RAG route |
| ChromaDB empty | Graceful error + setup instructions |
