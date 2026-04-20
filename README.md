# Lead-Gen Agentic Pipeline

Autonomous B2B sourcing agent built on **LangGraph**: discover companies from the web, verify registry context, crawl the site for buyer-specific signals, score with deterministic consensus, then enrich contacts with **Hunter.io**.

## Current status


| Area                 | Status                                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Four-node graph**  | Implemented: Discovery → Web crawler → Consensus → Enrichment (conditional routing).                                                                   |
| **Batch mode**       | Default entrypoint runs **Tavily fan-out discovery** → **concurrent** `run_sourcing_agent` per business → **CSV export** of consensus-qualified leads. |
| **Single-lead mode** | Supported via `run_sourcing_agent(...)` (library or custom script).                                                                                    |
| **LLM providers**    | **Grok (xAI)** default, or **Google Gemini**, or **local Ollama** (`LLM_PROVIDER`).                                                                    |
| **Mocks**            | `USE_MOCKS=true` for Tavily, Crawl4AI, Hunter, and LLM (no real API keys).                                                                             |
| **Tests**            | `tests/` currently has package scaffolding only; run `pytest` when you add tests.                                                                      |
| **Config**           | Loaded from `**.env.local`** (see [Configuration](#configuration)).                                                                                    |


## Architecture

```text
START
  ↓
Node 1 — Discovery (Tavily + structured LLM → RegistryVerification)
  ├→ active / unknown / not_found → Node 2 — Web crawler (Crawl4AI + LLM → WebsiteSignals)
  │                                    ↓
  │                                  Node 3 — Consensus (deterministic scoring, no LLM)
  │                                    ├→ passed (score ≥ threshold) → Node 4 — Enrichment (Hunter.io)
  │                                    │                                      ↓
  │                                    │                                    END
  │                                    └→ failed → END
  └→ inactive / suspended / dissolved → END
```

- **Node 1** also runs a **fallback Tavily + LLM** pass when no `official_website_url` is present in registry results (`WebsiteDiscovery`).
- **Node 2** uses investment_thesis in state to steer **dynamic** signal extraction (aligned with Node 3 scoring).
- **Node 3** uses fuzzy name/address alignment and thesis-driven signal points (`POINTS_PER_DETECTED_SIGNAL`, `MAX_BASE_SIGNAL_SCORE` in `src/nodes/consensus.py`). The pass threshold is the `CONSENSUS_THRESHOLD` constant in that file (60 by default); `CONSENSUS_SCORE_THRESHOLD` in `Config` is available for future wiring.

## Prerequisites

- **Python 3.10+** recommended (3.9 may work; match your environment).
- **Playwright** (for Crawl4AI): after `pip install`, run `playwright install` if crawls fail for missing browsers.
- API keys depend on provider and mocks (see below).

## Installation

```bash
git clone <your-repo-url>
cd "Lead-Gen Agentic Pipeline"

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# If Crawl4AI fails to launch browsers:
playwright install
```

## Configuration

The app loads environment variables from `**.env.local**` (`src/config.py` uses `load_dotenv('.env.local')`).

1. Copy the example file and rename:
  ```bash
   cp .env.example .env.local
  ```
2. Edit `**.env.local**`. Important variables:


| Variable                     | Purpose                                                        |
| ---------------------------- | -------------------------------------------------------------- |
| `LLM_PROVIDER`               | `grok` (default), `google`, or `ollama`.                       |
| `GROK_API_KEY`               | Required for Grok unless `USE_MOCKS=true`.                     |
| `GOOGLE_API_KEY`             | Required for Gemini when `LLM_PROVIDER=google`.                |
| `OLLAMA_BASE_URL`            | Default `http://localhost:11434` when using Ollama.            |
| `OLLAMA_MODEL`               | e.g. `llama3.1:8b` when using Ollama.                          |
| `LLM_MODEL`                  | Gemini model id when using Google (e.g. `gemini-2.5-flash`).   |
| `LLM_TEMPERATURE`            | Default `0` for reproducible extraction.                       |
| `TAVILY_API_KEY`             | Search (discovery + registry).                                 |
| `HUNTER_API_KEY`             | Domain email enrichment.                                       |
| `USE_MOCKS`                  | `true` / `false` — bypass real APIs when `true`.               |
| `MATCH_CONFIDENCE_THRESHOLD` | Name/address alignment tuning (used where wired in consensus). |


Grok calls use the OpenAI-compatible xAI endpoint with the model label configured in `src/services/llm_service.py`.

## Quick start

**Smoke test (mocks, single business):**

```bash
export USE_MOCKS=true
python quickstart.py
```

`quickstart.py` validates imports, config, graph build, and runs `**run_sourcing_agent**` once with mocks.

**Full batch pipeline (default `main`):**

```bash
# Real APIs: fill .env.local and set USE_MOCKS=false
python -m src.main
```

This path:

1. `**discover_businesses**` — Tavily multi-query fan-out (Ohio HVAC-style variants), dedupe, LLM extraction to a list of `{business_name, location}`.
2. `**run_batch_pipeline**` — `asyncio.gather` of `**run_sourcing_agent**` per discovered business (same graph as single-lead).
3. `**export_qualified_leads**` — Appends rows to `**qualified_leads.csv**` for leads with `passed_consensus=True`. Ensures a **header row** exists (migrates legacy headerless files once).

### CSV export columns


| Column        | Description                                            |
| ------------- | ------------------------------------------------------ |
| Business Name | Lead name                                              |
| Website       | Registry `official_website_url` if present, else `N/A` |
| Lead Score    | Final consensus score (0–100)                          |
| Contact Name  | Primary Hunter contact first + last, or `N/A`          |
| Contact Email | Primary contact email, or `N/A`                        |
| Job Title     | Primary contact title, or `N/A`                        |


## Programmatic usage

**Single lead:**

```python
import asyncio
from src.main import run_sourcing_agent

async def main():
    result = await run_sourcing_agent(
        query="Find HVAC distributors in Ohio without e-commerce",
        business_name="Example Company Corp",
        location="Cleveland, Ohio",
        website_url=None,
        investment_thesis="Optional buyer thesis for Node 2 signal design.",
    )
    print(result.lead_score, result.recommendation)

asyncio.run(main())
```

**Batch (custom driver):**

```python
import asyncio
from src.main import run_batch_pipeline, export_qualified_leads

async def main():
    results = await run_batch_pipeline(
        search_query="your Tavily seed query",
        extraction_instructions="Instructions for how many/what businesses to extract",
        investment_thesis="Thesis passed to each lead's Node 2",
        num_results_per_query=20,
    )
    export_qualified_leads(results, "qualified_leads.csv")

asyncio.run(main())
```

## Project layout

```text
src/
  main.py           # run_sourcing_agent, discover_businesses, run_batch_pipeline, CSV export, __main__ batch
  graph.py          # LangGraph: nodes + conditional edges
  config.py         # Env + Config.validate()
  models/           # Pydantic schemas + LeadState
  nodes/            # discovery, web_crawler, consensus, enrichment
  services/         # tavily, crawl4ai, hunter, llm (+ mocks)
tests/              # Placeholder for pytest suite
quickstart.py       # Mock single-lead verification
qualified_leads.csv # Generated/updated by export (committed optional)
verify-async.sh     # Dev checks for async/graph patterns (paths may need editing if you relocate the repo)
```

## Dependencies (high level)

Defined in `**requirements.txt**`: LangGraph, Pydantic v2, LangChain integrations (Google GenAI, Ollama, OpenAI-compatible for Grok), Crawl4AI, Tavily, httpx, Playwright, pytest stack, etc.

## Testing

```bash
export USE_MOCKS=true
pytest tests/ -v   # passes if no tests collected; add tests under tests/
```

## Extending

- **Thesis-driven signals:** Pass `investment_thesis` into `run_sourcing_agent` / `run_batch_pipeline`; adjust prompts in `src/nodes/web_crawler.py` and scoring caps in `src/nodes/consensus.py`.
- **Discovery fan-out:** Keyword/city variants and planning live in `**discover_businesses`** in `src/main.py`.
- **New LLM provider:** Extend `src/services/llm_service.py` and `src/config.py` / `.env.example`.

## Troubleshooting

- **Config not loading:** Ensure variables are in `**.env.local`**, not only `.env`.
- **Grok / Google / Ollama errors:** Match `LLM_PROVIDER` to keys you set; use `USE_MOCKS=true` to isolate graph logic.
- **Crawl failures:** Install browsers with `playwright install`; check firewall and target site blocking.
- **Empty CSV:** No rows appended if no lead in the batch has `passed_consensus`; check logs for consensus reject reasons.

## License / contact

Proprietary — update license and contact lines as appropriate for your organization.