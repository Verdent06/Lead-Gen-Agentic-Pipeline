# Lead-Gen Agentic Pipeline

Most agentic lead-gen systems fail the same way: a probabilistic LLM acts on unverified upstream data. One bad entity match, one hallucinated business attribute, one corrupted contact — and the downstream output is confidently wrong at scale.

This pipeline enforces **deterministic consensus gates at every state transition**. Raw web extractions cannot propagate forward unless they pass strict Pydantic schema validation and explicit source attribution checks. If the payload fails, the graph halts. Nothing malformed reaches the output layer.

Deployed for fragmented niche market sourcing across healthcare and private equity deal-flow pipelines.

---

## The Problem This Solves

Unstructured web data is inherently noisy. Headless scrapers hit anti-bot friction. Registry data conflicts with live site signals. LLMs infer missing fields rather than admitting uncertainty.

Standard pipelines silently merge these conflicting signals and pass malformed state forward. The result: enriched-looking leads that are factually wrong, delivered with full confidence.

This pipeline breaks that pattern with three hard constraints:

1. **No unverified state propagates.** Every node emits typed Pydantic artifacts. Downstream nodes are physically blocked from executing unless the upstream payload satisfies schema boundaries.
2. **No silent failures.** If a primary data source fails (Cloudflare block, broken DOM, empty registry result), the graph routes to a fallback rather than inferring missing data.
3. **No hallucinated actions.** The consensus gate runs deterministic scoring — no LLM involvement — before any lead is emitted. Score below threshold: pipeline halts.

---

## Architecture

```text
START
  ↓
Node 1 — Discovery
  Tavily multi-query fan-out → structured LLM extraction → RegistryVerification schema
  Fallback: if no official_website_url found → secondary Tavily + LLM WebsiteDiscovery pass
  ├→ active / unknown / not_found → Node 2
  └→ inactive / suspended / dissolved → END (hard halt)
  ↓
Node 2 — Web Crawler
  Crawl4AI headless scrape → LLM signal extraction → WebsiteSignals schema
  Steered by investment_thesis for dynamic, thesis-aligned signal targeting
  ↓
Node 3 — Consensus Gate (deterministic, no LLM)
  Fuzzy name/address alignment + thesis-driven signal scoring
  Hard threshold enforcement (default: 60/100)
  ├→ passed → Node 4
  └→ failed → END (hard halt)
  ↓
Node 4 — Enrichment
  Hunter.io domain lookup → contact name, email, title
  ↓
END → qualified_leads.csv
```

**Key design decision:** Node 3 runs zero LLM calls. Scoring is fully deterministic — fuzzy string alignment and explicit signal point tallying. This is intentional. A probabilistic model has no role in the pass/fail gate that controls outbound action.

---

## Source-Resilience Routing

Anti-bot friction (Cloudflare 403s, broken DOMs) is the most common cause of pipeline stall in production web ingestion. This pipeline handles it at the architecture level:

- **Primary:** Crawl4AI headless browser scrape
- **Fallback:** Autonomous reroute to Tavily cached SERP evidence when primary fails
- **Failure mode:** If both sources fail schema validation, the graph halts — it does not infer or hallucinate missing attributes

The pipeline never lets a data gap become a confident wrong answer.

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (state machine) |
| Schema enforcement | Pydantic v2 |
| Web discovery | Tavily API |
| Headless scraping | Crawl4AI + Playwright |
| Contact enrichment | Hunter.io |
| LLM providers | Grok (xAI) · Google Gemini · Ollama (local) |
| Async execution | asyncio + concurrent batch processing |

---

## Quick Start

**Smoke test with mocks (no API keys required):**

```bash
git clone <repo-url>
cd "Lead-Gen Agentic Pipeline"

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
playwright install

export USE_MOCKS=true
python quickstart.py
```

`quickstart.py` validates imports, config, graph build, and runs a full single-lead pass with mocked services. Use this to verify the orchestration layer before wiring real APIs.

**Full batch pipeline:**

```bash
cp .env.example .env.local
# Fill in API keys in .env.local
python -m src.main
```

Runs Tavily fan-out discovery → concurrent `run_sourcing_agent` per business → CSV export of consensus-qualified leads.

---

## Configuration

All environment variables load from `.env.local` (`src/config.py`).

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `grok` (default) · `google` · `ollama` |
| `GROK_API_KEY` | Required for Grok unless `USE_MOCKS=true` |
| `GOOGLE_API_KEY` | Required when `LLM_PROVIDER=google` |
| `OLLAMA_BASE_URL` | Default `http://localhost:11434` |
| `OLLAMA_MODEL` | e.g. `llama3.1:8b` |
| `LLM_MODEL` | Gemini model ID e.g. `gemini-2.5-flash` |
| `LLM_TEMPERATURE` | Default `0` — deterministic extraction |
| `TAVILY_API_KEY` | Discovery + registry search |
| `HUNTER_API_KEY` | Contact enrichment |
| `USE_MOCKS` | `true` bypasses all external APIs |
| `MATCH_CONFIDENCE_THRESHOLD` | Fuzzy name/address alignment tuning |

---

## Programmatic Usage

**Single lead:**

```python
import asyncio
from src.main import run_sourcing_agent

async def main():
    result = await run_sourcing_agent(
        query="HVAC distributors in Ohio without e-commerce",
        business_name="Example Company Corp",
        location="Cleveland, Ohio",
        website_url=None,
        investment_thesis="Targeting owner-operated distributors with no digital sales channel.",
    )
    print(result.lead_score, result.recommendation)

asyncio.run(main())
```

**Batch pipeline:**

```python
import asyncio
from src.main import run_batch_pipeline, export_qualified_leads

async def main():
    results = await run_batch_pipeline(
        search_query="independent HVAC distributors Midwest",
        extraction_instructions="Extract up to 20 businesses with name and location.",
        investment_thesis="Owner-operated, no e-commerce, under 50 employees.",
        num_results_per_query=20,
    )
    export_qualified_leads(results, "qualified_leads.csv")

asyncio.run(main())
```

---

## Output Schema

Consensus-qualified leads are exported to `qualified_leads.csv`:

| Column | Description |
|---|---|
| Business Name | Verified business name |
| Website | Registry URL if found, else `N/A` |
| Lead Score | Deterministic consensus score (0–100) |
| Contact Name | Primary Hunter.io contact |
| Contact Email | Primary contact email |
| Job Title | Primary contact title |

Only leads with `passed_consensus=True` are written. Every row in the output file has cleared the deterministic gate.

---

## Project Structure

```text
src/
  main.py           # run_sourcing_agent, discover_businesses, run_batch_pipeline, CSV export
  graph.py          # LangGraph graph: nodes + conditional edges
  config.py         # Env loading + Config.validate()
  models/           # Pydantic schemas + LeadState
  nodes/            # discovery, web_crawler, consensus, enrichment
  services/         # tavily, crawl4ai, hunter, llm (+ mock implementations)
tests/
quickstart.py       # Mock single-lead smoke test
qualified_leads.csv # Generated by export pipeline
```

---

## Extending

**Thesis-driven signal targeting:** Pass `investment_thesis` into `run_sourcing_agent` or `run_batch_pipeline`. Node 2 uses it to steer dynamic signal extraction. Node 3 scoring weights are configurable in `src/nodes/consensus.py` (`POINTS_PER_DETECTED_SIGNAL`, `MAX_BASE_SIGNAL_SCORE`, `CONSENSUS_THRESHOLD`).

**New LLM provider:** Extend `src/services/llm_service.py` and update `src/config.py` + `.env.example`.

**Discovery fan-out:** Keyword and city variants live in `discover_businesses` in `src/main.py`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Config not loading | Variables must be in `.env.local`, not `.env` |
| LLM auth errors | Match `LLM_PROVIDER` to the key you set; use `USE_MOCKS=true` to isolate |
| Crawl failures | Run `playwright install`; check firewall and target site bot protection |
| Empty CSV output | No leads cleared consensus — check logs for scoring reject reasons |

---

## License

MIT License — © 2025 Vedant Desai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the condition that the above copyright notice appears in all copies.
