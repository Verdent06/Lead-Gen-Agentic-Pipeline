"""
LangSmith evaluation runner for run_sourcing_agent against the golden dataset.

Creates (or reuses) the "HVAC_Lead_Gen_Golden_20" LangSmith dataset, runs the
pipeline with Hunter.io mocked, and scores results with deterministic + LLM judges.

Run from project root:

    python -m tests.langsmith_eval
    python -m tests.langsmith_eval --dataset tests/eval_dataset.json
    python -m tests.langsmith_eval --upload-results false

    # Test display formatting only — no pipeline runs, no API credits used:
    python -m tests.langsmith_eval --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.evaluation import aevaluate
from pydantic import BaseModel, Field

from src.cli.discover import run_sourcing_agent
from src.core.config import Config
from src.models.schemas import FinalLeadOutput

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "tests" / "eval_dataset.json"
LANGSMITH_DATASET_NAME = "HVAC_Lead_Gen_Golden_20"
MARKDOWN_CHAR_LIMIT = 24_000

# ---------------------------------------------------------------------------
# Hunter mock (Node 4) — identical to tests/evaluate_pipeline.py
# ---------------------------------------------------------------------------


def _build_mock_hunter_contacts(domain: str) -> list[dict[str, Any]]:
    safe_domain = domain.replace("/", "_") or "example.com"
    return [
        {
            "first_name": "Eval",
            "last_name": "Owner",
            "email": f"eval.owner@{safe_domain}",
            "email_confidence": 0.97,
            "job_title": "Owner/CEO",
            "department": "Management",
            "linkedin_profile": None,
            "phone": None,
        },
        {
            "first_name": "Eval",
            "last_name": "Ops",
            "email": f"eval.ops@{safe_domain}",
            "email_confidence": 0.91,
            "job_title": "Operations Manager",
            "department": "Operations",
            "linkedin_profile": None,
            "phone": None,
        },
    ]


def _make_mock_hunter_service() -> AsyncMock:
    mock_service = AsyncMock()
    mock_service.find_contacts = AsyncMock(
        side_effect=lambda domain, company_name=None, owner_name=None: _build_mock_hunter_contacts(
            domain
        )
    )
    return mock_service


async def _mock_get_hunter_service() -> AsyncMock:
    return _make_mock_hunter_service()


# Patch both import sites so the mock survives LangSmith concurrent workers.
_HUNTER_PATCH_TARGETS = (
    "src.agent.nodes.enrichment.get_hunter_service",
    "src.services.hunter_service.get_hunter_service",
)
_hunter_patchers: list[Any] = []


def _start_hunter_mock() -> None:
    """Install Hunter mock for the full eval session (avoids concurrent patch teardown races)."""
    if _hunter_patchers:
        return
    for target in _HUNTER_PATCH_TARGETS:
        patcher = patch(target, new=_mock_get_hunter_service)
        patcher.start()
        _hunter_patchers.append(patcher)


def _stop_hunter_mock() -> None:
    while _hunter_patchers:
        _hunter_patchers.pop().stop()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_local_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset must be a non-empty JSON array: {path}")
    return data


def _dataset_exists(client: Client, dataset_name: str) -> bool:
    return any(True for _ in client.list_datasets(dataset_name=dataset_name))


def ensure_langsmith_dataset(
    client: Client,
    cases: list[dict[str, Any]],
    dataset_name: str = LANGSMITH_DATASET_NAME,
) -> str:
    """Create the LangSmith dataset if missing; return dataset name."""
    if _dataset_exists(client, dataset_name):
        logging.info("LangSmith dataset %r already exists — skipping upload.", dataset_name)
        return dataset_name

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=(
            "20-case golden dataset for independent B2B HVAC distributor lead-gen. "
            "Includes adversarial negatives and localized pass cases."
        ),
    )

    examples = [
        {
            "inputs": case["inputs"],
            "outputs": case["expected_outputs"],
            "metadata": {
                "case_id": case.get("case_id", ""),
                "description": case.get("description", ""),
            },
        }
        for case in cases
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    logging.info(
        "Created LangSmith dataset %r with %d examples.",
        dataset_name,
        len(examples),
    )
    return dataset_name


# ---------------------------------------------------------------------------
# Target function (pipeline wrapper)
# ---------------------------------------------------------------------------


def _lead_output_to_dict(output: FinalLeadOutput) -> dict[str, Any]:
    website_signals = output.website_signals
    is_target_industry: Optional[bool] = None
    industry_evidence = ""
    if website_signals is not None:
        is_target_industry = website_signals.is_target_industry
        industry_evidence = website_signals.industry_evidence or ""

    return {
        "consensus_passed": output.passed_consensus,
        "lead_score": output.lead_score,
        "is_target_industry": is_target_industry,
        "website_markdown": output.raw_content or "",
        "industry_evidence": industry_evidence,
        "business_name": output.business_name,
        "errors_encountered": output.errors_encountered,
    }


async def langsmith_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    LangSmith target: invoke run_sourcing_agent with Hunter.io patched out.

    Returns a flat dict aligned with golden-dataset expected_outputs keys plus
    trace fields needed by custom evaluators (website_markdown, industry_evidence).
    """
    # Re-assert patch inside each worker invocation (LangSmith may spawn threads).
    with patch.multiple(
        "src.agent.nodes.enrichment",
        get_hunter_service=_mock_get_hunter_service,
    ), patch.multiple(
        "src.services.hunter_service",
        get_hunter_service=_mock_get_hunter_service,
    ):
        output = await run_sourcing_agent(
            query=inputs["query"],
            business_name=inputs["business_name"],
            location=inputs["location"],
            website_url=inputs.get("website_url"),
            investment_thesis=inputs.get("investment_thesis"),
        )
    return _lead_output_to_dict(output)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def consensus_passed_match(outputs: dict, reference_outputs: dict) -> bool:
    """Deterministic: actual consensus_passed == expected expected_consensus_passed."""
    actual = bool(outputs.get("consensus_passed"))
    expected = bool(reference_outputs.get("expected_consensus_passed"))
    return actual == expected


class FaithfulnessVerdict(BaseModel):
    """LLM judge output: 1 = evidence strictly supported by markdown, 0 = not."""

    score: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if industry_evidence is strictly factually supported by website_markdown; else 0.",
    )
    rationale: str = Field(
        default="",
        description="Brief justification for the score.",
    )


_FAITHFULNESS_SYSTEM = """You are a strict factual-consistency judge for web-extraction QA.

You will receive:
1. website_markdown — raw text crawled from a company website.
2. industry_evidence — a claim produced by another model about what the site says regarding industry fit.

Task: Decide whether industry_evidence is STRICTLY factually supported by website_markdown.

Rules:
- Score 1 ONLY if every substantive claim in industry_evidence appears in or is directly entailed by the markdown.
- Score 0 if industry_evidence adds facts not present, misquotes, over-interprets, or contradicts the markdown.
- Ignore whether the business is actually an HVAC distributor or whether the pipeline should pass/fail.
- If website_markdown is empty or industry_evidence is empty, score 0.
- Paraphrases are acceptable only when meaning is preserved without new facts."""


def _create_faithfulness_judge():
    """Prefer OpenAI for the judge; fall back to Google Gemini."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        model = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0, api_key=openai_key)

    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        model = os.getenv("EVAL_JUDGE_MODEL", Config.LLM_MODEL)
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=google_key,
        )

    raise ValueError(
        "Faithfulness judge requires OPENAI_API_KEY or GOOGLE_API_KEY/GEMINI_API_KEY."
    )


_judge_llm = None


def _get_faithfulness_judge():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = _create_faithfulness_judge().with_structured_output(FaithfulnessVerdict)
    return _judge_llm


async def industry_evidence_faithfulness(outputs: dict) -> dict[str, Any]:
    """
    LLM-as-judge: grade whether Node 2 industry_evidence is grounded in website_markdown.

    Returns score 1 (faithful) or 0 (unfaithful / insufficient data).
    """
    markdown = (outputs.get("website_markdown") or "").strip()
    evidence = (outputs.get("industry_evidence") or "").strip()

    if not markdown or not evidence:
        return {
            "key": "industry_evidence_faithfulness",
            "score": 0,
            "comment": "Missing website_markdown or industry_evidence.",
        }

    truncated = markdown[:MARKDOWN_CHAR_LIMIT]
    if len(markdown) > MARKDOWN_CHAR_LIMIT:
        truncated += "\n\n[... markdown truncated for judge context ...]"

    user_content = (
        f"=== WEBSITE MARKDOWN ===\n{truncated}\n\n"
        f"=== INDUSTRY EVIDENCE ===\n{evidence}"
    )

    judge = _get_faithfulness_judge()
    verdict: FaithfulnessVerdict = await judge.ainvoke(
        [
            SystemMessage(content=_FAITHFULNESS_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {
        "key": "industry_evidence_faithfulness",
        "score": verdict.score,
        "comment": verdict.rationale[:500] if verdict.rationale else "",
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpcore", "crawl4ai", "playwright"):
        logging.getLogger(name).setLevel(logging.WARNING)


async def run_langsmith_evaluation(
    dataset_path: Path,
    experiment_prefix: str,
    upload_results: bool,
    max_concurrency: int,
) -> Any:
    _configure_logging(quiet=True)
    cases = _load_local_dataset(dataset_path)

    client = Client()
    dataset_name = ensure_langsmith_dataset(client, cases)

    logging.info("Starting LangSmith evaluation on dataset %r", dataset_name)
    logging.info("Hunter.io mocked via unittest.mock — no enrichment API credits used.")

    _start_hunter_mock()
    try:
        experiment = await aevaluate(
            langsmith_target,
            data=dataset_name,
            evaluators=[consensus_passed_match, industry_evidence_faithfulness],
            experiment_prefix=experiment_prefix,
            max_concurrency=max_concurrency,
            upload_results=upload_results,
            client=client,
        )
        await experiment.wait()
    finally:
        _stop_hunter_mock()

    return experiment


# ---------------------------------------------------------------------------
# Terminal reporting
# ---------------------------------------------------------------------------

# Column definitions: (dict_key, header_label, width)
_COLUMNS = [
    ("idx",          "#",        3),
    ("business",     "Business", 30),
    ("exp_cons",     "Exp",       5),
    ("act_cons",     "Act",       5),
    ("cons_match",   "Match",     5),
    ("faithfulness", "Faith",     5),
    ("lead_score",   "Score",     5),
    ("error",        "Err",       3),
]


def _feedback_score(row: Any, key: str) -> Any:
    """Extract a feedback score from a raw LangSmith result row."""
    for result in row.get("evaluation_results", {}).get("results", []):
        if result.key == key:
            return result.score if result.score is not None else result.value
    return None


def _extract_row(idx: int, raw_row: Any) -> dict[str, Any]:
    """
    Convert a raw LangSmith result row (dict with 'example', 'run', etc.)
    into a flat display dict consumed by _print_results_table.
    """
    inputs    = raw_row["example"].inputs  or {}
    outputs   = raw_row["run"].outputs     or {}
    reference = raw_row["example"].outputs or {}

    return {
        "idx":          idx,
        "business":     str(inputs.get("business_name", ""))[:30],
        "exp_cons":     reference.get("expected_consensus_passed"),
        "act_cons":     outputs.get("consensus_passed"),
        "cons_match":   _feedback_score(raw_row, "consensus_passed_match"),
        "faithfulness": _feedback_score(raw_row, "industry_evidence_faithfulness"),
        "lead_score":   outputs.get("lead_score"),
        "error":        "yes" if raw_row["run"].error else "no",
    }


def _print_results_table(rows: list[dict[str, Any]]) -> None:
    """
    Render pre-extracted display dicts as a fixed-width terminal table.
    Receives the output of [_extract_row(i, r) for i, r in enumerate(raw_results, 1)].
    """
    # Build header and separator
    header = "  ".join(f"{label:>{width}}" for _, label, width in _COLUMNS)
    sep    = "-" * len(header)
    print(header)
    print(sep)

    match_hits: int = 0
    faith_scores: list[float] = []

    for row in rows:
        cells = []
        for key, _, width in _COLUMNS:
            val = row.get(key)
            cell = str(val) if val is not None else "-"
            cells.append(f"{cell:>{width}}")
        print("  ".join(cells))

        if row.get("cons_match"):
            match_hits += 1
        faith = row.get("faithfulness")
        if isinstance(faith, (int, float)):
            faith_scores.append(float(faith))

    print(sep)
    total = len(rows)
    if total:
        avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
        print(
            f"  consensus_match {match_hits}/{total} ({match_hits / total * 100:.1f}%)  |  "
            f"faithfulness avg {avg_faith:.2f}"
        )


def _experiment_results_list(experiment: Any) -> list[Any]:
    """
    Materialize experiment rows for terminal reporting.

    AsyncExperimentResults is async-iterable only; after wait(), rows live on
    `_results`. Sync ExperimentResults supports list() directly.
    """
    if hasattr(experiment, "_results"):
        return list(experiment._results)
    return list(experiment)


def _print_experiment_summary(experiment: Any) -> None:
    """
    Print a clean, Claude Code-readable summary of experiment results.
    No pandas — avoids the website_markdown column explosion and venv issues.
    """
    raw_results = _experiment_results_list(experiment)
    total = len(raw_results)

    print("\n" + "=" * 72)
    print("  LANGSMITH EVALUATION SUMMARY")
    print("=" * 72)
    if hasattr(experiment, "experiment_name"):
        print(f"  Experiment   : {experiment.experiment_name}")
    print(f"  Cases        : {total}")

    if total == 0:
        print("  No results returned.")
        print("=" * 72 + "\n")
        return

    # Pre-extract all rows once — shared by aggregate stats and the table
    rows = [_extract_row(i, r) for i, r in enumerate(raw_results, 1)]

    match_hits  = sum(1 for r in rows if r["cons_match"])
    faith_vals  = [r["faithfulness"] for r in rows if isinstance(r["faithfulness"], (int, float))]
    avg_faith   = sum(faith_vals) / len(faith_vals) if faith_vals else 0.0

    print(f"  consensus_passed_match              : {match_hits}/{total} ({match_hits / total * 100:.1f}%)")
    print(f"  industry_evidence_faithfulness (avg): {avg_faith:.2f}")
    print("=" * 72)
    print()

    # Pass pre-extracted dicts — NOT raw rows
    _print_results_table(rows)

    print("\n" + "=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Dry-run: test the display layer without running the pipeline
# ---------------------------------------------------------------------------

def _make_dry_run_rows(n: int = 20) -> list[dict[str, Any]]:
    """
    Generate n synthetic display-ready rows for --dry-run mode.
    Covers the full range of pass/fail/error combinations so the table
    rendering can be verified without any API calls.
    """
    import random
    random.seed(42)

    businesses = [
        "Arctic Air Distributors",  "Sunbelt HVAC Supply Co",
        "Great Lakes Comfort Systems", "Desert Breeze Equipment",
        "Coastal Climate Solutions", "Mountain Peak Air Systems",
        "Valley Heating & Cooling",  "Metro HVAC Wholesale",
        "Lakeside Equipment Corp",   "Ridgeline Climate Supply",
        "Heartland HVAC Distribs",   "Pacific Northwest Air",
        "Gulf Coast Comfort Equip",  "Blue Ridge HVAC Supply",
        "Prairie Wind Systems Inc",  "Cascade Climate Controls",
        "Tidewater HVAC Wholesale",  "Rockland Equipment Co",
        "Pinecrest Air Solutions",   "Clearwater HVAC Supply",
    ]

    rows = []
    for i in range(1, n + 1):
        exp  = random.choice([True, False])
        act  = exp if random.random() > 0.3 else (not exp)
        faith = round(random.uniform(0, 1)) if random.random() > 0.1 else None
        rows.append({
            "idx":          i,
            "business":     businesses[(i - 1) % len(businesses)][:30],
            "exp_cons":     exp,
            "act_cons":     act,
            "cons_match":   act == exp,
            "faithfulness": faith,
            "lead_score":   round(random.uniform(0.3, 0.95), 2) if act else None,
            "error":        "yes" if random.random() < 0.05 else "no",
        })
    return rows


def _run_dry_run() -> None:
    """Print a synthetic summary to validate display formatting — zero API calls."""
    print("\n" + "=" * 72)
    print("  LANGSMITH EVALUATION SUMMARY  [DRY RUN — synthetic data]")
    print("=" * 72)
    print(f"  Experiment   : hvac-lead-gen-golden-20-dryrun")

    rows = _make_dry_run_rows(20)
    total = len(rows)
    print(f"  Cases        : {total}")

    match_hits = sum(1 for r in rows if r["cons_match"])
    faith_vals = [r["faithfulness"] for r in rows if isinstance(r["faithfulness"], (int, float))]
    avg_faith  = sum(faith_vals) / len(faith_vals) if faith_vals else 0.0

    print(f"  consensus_passed_match              : {match_hits}/{total} ({match_hits / total * 100:.1f}%)")
    print(f"  industry_evidence_faithfulness (avg): {avg_faith:.2f}")
    print("=" * 72)
    print()
    _print_results_table(rows)
    print("\n" + "=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluation for run_sourcing_agent against the golden dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Local JSON golden dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="hvac-lead-gen-golden-20",
        help="LangSmith experiment name prefix.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=2,
        help="Max parallel pipeline runs (default: 2).",
    )
    parser.add_argument(
        "--upload-results",
        choices=("true", "false"),
        default="true",
        help="Upload traces/results to LangSmith (default: true).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the pipeline entirely and print a synthetic results table. "
            "Use this to validate display formatting without spending API credits."
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        _run_dry_run()
        return

    upload = args.upload_results.lower() == "true"

    try:
        experiment = asyncio.run(
            run_langsmith_evaluation(
                dataset_path=args.dataset.resolve(),
                experiment_prefix=args.experiment_prefix,
                upload_results=upload,
                max_concurrency=args.max_concurrency,
            )
        )
    except Exception as exc:
        print(f"LangSmith evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_experiment_summary(experiment)


if __name__ == "__main__":
    main()