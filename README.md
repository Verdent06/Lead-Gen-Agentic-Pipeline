# Lead-Gen Agentic Pipeline

Most agentic lead-gen systems fail the same way: a probabilistic LLM acts on unverified upstream data. One bad entity match, one hallucinated business attribute, one corrupted contact — and the downstream output is confidently wrong at scale.

This pipeline enforces **deterministic consensus gates at every state transition**. Raw web extractions cannot propagate forward unless they pass strict Pydantic schema validation and explicit source attribution checks.

Deployed for fragmented niche market sourcing across healthcare, private equity deal-flow pipelines, and B2B enterprise software consulting.

---

## The Dual-Engine Architecture

This repository operates via two distinct execution engines that share the same underlying LangGraph state machine, database, and API wrappers.

### Engine A: The Discovery Engine (`src/cli/discover.py`)

Used for finding _net-new_ companies based on a natural language investment thesis.

- **Map:** Takes a query (e.g., "HVAC distributors in Ohio"). Uses `Tavily` to fan out Google searches and an LLM to parse SERPs into a list of target companies.
- **Graph Flow:** `Discovery (Tavily)` → `Web Crawler` → `Consensus (Strict)` → `Enrichment`.
- **Gate:** If a company's web signals do not score above the deterministic threshold (default 60/100) based on the investment thesis, it is hard-dropped.

### Engine B: The Targeted Enrichment Engine (`src/cli/enrich.py`)

Used for enriching a _known_ list of targets (e.g., extracting customers from a competitor's directory).

- **Map:** Takes a specific target URL. Uses `Crawl4AI` and an LLM to extract every exact company name listed on the page.
- **Graph Flow:** `Discovery (Tavily)` → `Web Crawler` → `Consensus (Ignored)` → `Enrichment`.
- **Gate:** Bypasses strict consensus because the targets are already pre-qualified. Saves all extracted payloads and Hunter.io contacts directly to the PostgreSQL database via asynchronous fan-out.

---

## Stack

| Layer                  | Technology                                  |
| ---------------------- | ------------------------------------------- |
| **REST API**           | FastAPI + Uvicorn                           |
| **Orchestration**      | LangGraph (State Machine)                   |
| **Database**           | PostgreSQL + asyncpg                        |
| **Schema Enforcement** | Pydantic v2                                 |
| **Web Discovery**      | Tavily API                                  |
| **Headless Scraping**  | Crawl4AI + Playwright                       |
| **Contact Enrichment** | Hunter.io                                   |
| **LLM Providers**      | Grok (xAI) · Google Gemini · Ollama (Local) |
| **Async Execution**    | asyncio + concurrent batch processing       |

---

## Project Structure (Domain-Driven Layout)

```text
lead-gen-agentic-pipeline/
├── infra/                  # Database migrations and Docker configurations
│   └── migrations/
│       └── 001_add_contact_columns.sql
├── tests/                  # Pytest directory
├── docker-compose.yml      # Spins up PostgreSQL / pgvector
├── requirements.txt
├── README.md
└── src/
    ├── core/
    │   └── config.py       # Global environment validation
    ├── api/
    │   └── server.py       # FastAPI application and REST endpoints
    ├── cli/
    │   ├── discover.py     # Engine A Entry Point (Net-new sourcing)
    │   └── enrich.py       # Engine B Entry Point (Directory extraction)
    ├── agent/
    │   ├── graph.py        # LangGraph node routing and edge logic
    │   ├── state.py        # LeadState TypedDict definition
    │   └── nodes/          # Executable LangGraph nodes
    │       ├── discovery.py
    │       ├── web_crawler.py
    │       ├── consensus.py
    │       └── enrichment.py
    ├── services/           # External API Wrappers
    │   ├── db_service.py
    │   ├── llm_service.py
    │   ├── crawl4ai_service.py
    │   ├── hunter_service.py
    │   └── tavily_service.py
    └── models/
        └── schemas.py      # Pydantic data validation models
```

---

## Quick Start & Usage

### 1. Environment Setup

```bash
git clone <repo-url>
cd lead-gen-agentic-pipeline

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
playwright install
```

Copy `.env.example` to `.env.local` and fill in your API keys (Tavily, Hunter, LLM Provider).

### 2. Infrastructure Setup

Start the local PostgreSQL database:

```bash
docker-compose up -d db
```

### 3. Execution Commands

**Run the REST API Server:**

```bash
uvicorn src.api.server:app --reload
```

Access the interactive Swagger UI at `http://localhost:8000/docs` to trigger agentic workflows via HTTP.

**Run Engine A (Discovery CLI):**

```bash
python -m src.cli.discover
```

**Run Engine B (Targeted Enrichment CLI):**

```bash
python -m src.cli.enrich
```

---

## Configuration

All environment variables load from `.env.local` (`src/core/config.py`).

| Variable          | Purpose                                             |
| ----------------- | --------------------------------------------------- |
| `LLM_PROVIDER`    | `grok` (default) · `google` · `ollama`              |
| `GROK_API_KEY`    | Required for Grok unless `USE_MOCKS=true`           |
| `OLLAMA_BASE_URL` | Default `http://localhost:11434`                    |
| `TAVILY_API_KEY`  | Discovery + registry search                         |
| `HUNTER_API_KEY`  | Contact enrichment                                  |
| `DB_URL`          | PostgreSQL connection string                        |
| `USE_MOCKS`       | `true` bypasses all external APIs for local testing |

---

## Troubleshooting

| Symptom                 | Fix                                                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `ImportError`           | Ensure you run scripts as modules from the root (e.g., `python -m src.cli.enrich`). Do not execute files inside `src/` directly. |
| DB Connection Refused   | Ensure Docker is running and `docker-compose up -d db` was executed.                                                             |
| Empty Hunter.io Payload | Ensure `TARGET_PERSONAS` in the graph execution aligns with the `department` parameter sent to the Hunter.io API.                |

---

## License

MIT License — © 2026 Vedant Desai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the condition that the above copyright notice appears in all copies.
