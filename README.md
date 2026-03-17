"""
Autonomous Sourcing Agent for B2B Lead Generation

A production-ready LangGraph pipeline that replaces static B2B databases (like ZoomInfo)
for high-ticket buyers. Uses deterministic orchestration, LLM-powered extraction, and
triangulated consensus scoring to identify and enrich unstructured B2B leads.

Architecture:
  Node 1: Discovery & State Registry Check (Tavily)
  Node 2: Dynamic Website Crawler & Signal Extraction (Crawl4AI + Gemini)
  Node 3: Triangulated Consensus & Scoring (Deterministic Python)
  Node 4: Enrichment with Contacts (Hunter.io)

Tech Stack:
  - LangGraph: State orchestration and graph routing
  - Pydantic v2: Strict schema validation
  - Google Gemini: LLM with structured output
  - Crawl4AI: Local headless Playwright for DOM-to-Markdown conversion
  - Tavily: Web search for state registries
  - Hunter.io: Email/contact enrichment
"""

# Project Initialization

## Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- API Keys: Google Gemini, Tavily, Hunter.io

## Installation

1. Clone the repository and navigate to the project:
```bash
cd /Users/vedantdesai/Projects/Lead-Gen\ Agentic\ Pipeline
```

2. Create and activate virtual environment:
```bash
python3 -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys:
# GOOGLE_API_KEY=your-key-here
# TAVILY_API_KEY=your-key-here
# HUNTER_API_KEY=your-key-here
```

## Quick Start

### Run with Mock Data (No API Keys Required)

```bash
export USE_MOCKS=true
python -m src.main
```

### Run with Real APIs

```bash
# Ensure .env is populated with real API keys
python -m src.main
```

### Use as a Library

```python
import asyncio
from src.main import run_sourcing_agent

async def main():
    result = await run_sourcing_agent(
        query="Find HVAC distributors in Ohio without e-commerce",
        business_name="Smith HVAC Distributors",
        location="Cleveland, Ohio",
    )
    print(f"Lead Score: {result.lead_score}/100")
    print(f"Recommendation: {result.recommendation}")

asyncio.run(main())
```

## Architecture Overview

### Graph Flow

```
START
  ↓
NODE 1: Discovery (Registry Check via Tavily)
  ├→ Active → NODE 2: Web Crawler (Crawl4AI + LLM Signal Extraction)
  │              ↓
  │            NODE 3: Consensus (Deterministic Python - NO LLM)
  │              ├→ Score ≥ 70 → NODE 4: Enrichment (Hunter.io)
  │              │                ↓
  │              │              END (Return Lead)
  │              │
  │              └→ Score < 70 → END (Reject Lead)
  │
  └→ Inactive → END (Skip Pipeline)
```

### Node Descriptions

#### Node 1: Discovery & State Registry Check
- **Purpose**: Verify business legitimacy and active status
- **Service**: Tavily API (web search)
- **Process**:
  1. Search for official state licensing registries
  2. Use Gemini to extract structured RegistryVerification
  3. Validate: Business must be "active"
- **Routing**:
  - Active → Continue to Node 2
  - Inactive/Not Found → END (save tokens, skip waste)
- **Pydantic Output**: `RegistryVerification`

#### Node 2: Dynamic Website Crawler & Signal Extraction
- **Purpose**: Extract hidden business signals from website
- **Service**: Crawl4AI (local Playwright) + Gemini LLM
- **Process**:
  1. Crawl business website (Crawl4AI converts DOM to Markdown)
  2. Use Gemini to extract signals from Markdown with strict Pydantic validation
  3. Extract:
     - E-commerce presence (Shopify, WooCommerce, custom, none)
     - Legacy software mentions (Flash, old ASP.NET, outdated tech)
     - Succession planning signals (family members, ownership transitions)
     - Owner retirement mentions (age, retirement timeline)
     - Contact information and team size
- **Pydantic Output**: `WebsiteSignals` (with per-signal confidence scores and evidence)

#### Node 3: Triangulated Consensus & Deterministic Scoring
- **Purpose**: Validate registry vs. website data; calculate final lead score
- **Type**: **PURE DETERMINISTIC PYTHON - NO LLM CALLS**
- **Process**:
  1. Fuzzy-match business names (registry vs. website)
  2. Fuzzy-match addresses (registry vs. website)
  3. Detect conflicts:
     - Name match < 0.70 → Reject
     - Address match < 0.65 → Reject (or warn)
  4. Calculate `final_lead_score` (0-100):
     - Base: Signal scoring
       - No e-commerce: +25 pts
       - Legacy software: +20 pts
       - Succession planning: +20 pts
       - Owner retirement signals: +25 pts
     - Bonus: Match quality (0-20 pts)
       - Excellent match (name ≥0.95, address ≥0.90): +20 pts
       - Good match (name ≥0.85, address ≥0.75): +10 pts
  5. Routing:
     - Score ≥ 70 AND no conflicts → Proceed to Node 4
     - Score < 70 OR conflicts → END (reject lead)
- **Pydantic Output**: `ConsensusResult` (with detailed scoring breakdown)

#### Node 4: Enrichment
- **Purpose**: Find contact information for owner/decision makers
- **Service**: Hunter.io API
- **Process**:
  1. Search Hunter.io for domain contacts
  2. Identify primary contact (CEO/Owner priority)
  3. Return list of enriched contacts with emails, job titles, LinkedIn
- **Pydantic Output**: `List[HunterContact]`

### State Management

The `LeadState` TypedDict is passed through all nodes. Key features:

- **Annotated List Reducers**: `execution_log` and `errors_encountered` use `operator.add` reducer to **append** (not overwrite) entries
- **Comprehensive Tracking**: 30+ fields for input parameters, intermediate outputs, metadata, and execution logs
- **Type Safety**: All fields typed for IDE autocomplete and validation

### Pydantic Schemas

All LLM outputs are validated against strict Pydantic models:

- `RegistryVerification`: Registry data from Node 1
- `SignalCategory`: Individual signal with confidence and evidence
- `WebsiteSignals`: Website signals from Node 2
- `ConsensusResult`: Scoring breakdown from Node 3
- `HunterContact`: Contact information from Node 4
- `FinalLeadOutput`: Complete lead object returned to user

Each schema enforces:
- Type hints and Optional fields
- Confidence scores (0.0-1.0) for probabilistic data
- Evidence strings for audit trails
- Validators for normalization (e.g., lowercase status fields)

## Configuration

All configuration via environment variables in `.env`:

```env
# LLM Configuration
GOOGLE_API_KEY=your-key
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0

# API Keys
TAVILY_API_KEY=your-key
HUNTER_API_KEY=your-key

# Thresholds
CONSENSUS_SCORE_THRESHOLD=70
MATCH_CONFIDENCE_THRESHOLD=0.85

# Feature Flags
USE_MOCKS=false
```

## Extending the Pipeline

### Add Custom Signals in Node 2

Edit `src/models/schemas.py` → `WebsiteSignals`:
```python
class WebsiteSignals(BaseModel):
    # ... existing signals ...
    
    my_custom_signal: SignalCategory = Field(
        ..., description="My custom signal description"
    )
```

Then update the Node 2 LLM prompt in `src/nodes/web_crawler.py` to extract the new signal.

### Adjust Node 3 Scoring

Edit `src/nodes/consensus.py` → `SIGNAL_SCORES`:
```python
SIGNAL_SCORES = {
    "no_ecommerce": 25,
    "legacy_software": 20,
    "my_custom_signal": 15,  # Add new signal
}
```

### Use Different LLM

Edit `src/services/llm_service.py` to swap Gemini for Claude, GPT-4o, or other:
```python
# Change from ChatGoogleGenerativeAI to ChatAnthropic, ChatOpenAI, etc.
```

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run with mock data (no API calls):
```bash
export USE_MOCKS=true
python -m src.main
```

## Production Deployment

1. **Environment Secrets**: Use secure secret management (e.g., AWS Secrets Manager)
2. **Async Runtime**: Deploy with async-capable ASGI server (e.g., Uvicorn, Gunicorn with uvicorn worker)
3. **Rate Limiting**: Add request throttling for API calls
4. **Caching**: Cache Tavily/Hunter results to reduce API costs
5. **Monitoring**: Log all pipeline executions for audit and debugging
6. **Error Handling**: Implement retry logic with exponential backoff for external APIs

## Architecture Principles

1. **No Hardcoded HTML Parsing**: All website data flows through Crawl4AI → Markdown → LLM
2. **Deterministic Consensus**: Node 3 is pure Python (no probabilistic LLM) for reproducibility
3. **Triangulated Validation**: Registry + Website data must align to avoid hallucinations
4. **Fail-Fast Routing**: Early termination if registry check fails (saves tokens/compute)
5. **Full Type Safety**: Pydantic models enforce schema contracts
6. **Async Throughout**: Every node and service is async-ready for high throughput
7. **Mock Support**: All external APIs have mock implementations for local development

## Troubleshooting

### No results from Tavily
- Ensure `TAVILY_API_KEY` is valid
- Check network connectivity
- Verify search query is specific enough

### LLM extraction returning None
- Check Gemini API key validity
- Ensure `LLM_TEMPERATURE=0` for deterministic output
- Review LLM response format (ensure valid JSON)

### Low lead scores
- Adjust `SIGNAL_SCORES` in `src/nodes/consensus.py`
- Lower `CONSENSUS_SCORE_THRESHOLD` in `.env`
- Review scoring logic: ensure signals align with your criteria

### Address/name matching failing
- Lower `MATCH_CONFIDENCE_THRESHOLD` in `.env`
- Update fuzzy matching logic in `consensus_node()`
- Manually review edge cases

## License

Proprietary - YC Startup

## Contact

Vedant Desai - vedant@company.com
"""

import logging

logger = logging.getLogger(__name__)
