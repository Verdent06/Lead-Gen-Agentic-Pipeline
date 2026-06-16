"""
Golden-dataset evaluation runner for run_sourcing_agent.

Mocks Hunter.io (Node 4) so evaluation does not consume API credits.
Run from project root:

    python -m tests.evaluate_pipeline
    python -m tests.evaluate_pipeline --dataset tests/eval_dataset.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

from src.cli.discover import run_sourcing_agent
from src.models.schemas import FinalLeadOutput, HunterContact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "tests" / "eval_dataset.json"

# ---------------------------------------------------------------------------
# Console styling (no extra dependencies)
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"{code}{text}{RESET}"


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Hunter mock (Node 4)
# ---------------------------------------------------------------------------


def _build_mock_hunter_contacts(domain: str) -> list[dict[str, Any]]:
    """Deterministic fake Hunter payload — never hits the network."""
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


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


@dataclass
class CriterionResult:
    name: str
    expected: Any
    actual: Any
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    description: str
    business_name: str
    passed: bool
    criteria: list[CriterionResult] = field(default_factory=list)
    lead_score: int = 0
    execution_time_seconds: Optional[float] = None
    errors: list[str] = field(default_factory=list)


def _actual_is_target_industry(output: FinalLeadOutput) -> Optional[bool]:
    if output.website_signals is None:
        return None
    return output.website_signals.is_target_industry


def _evaluate_case_output(
    output: FinalLeadOutput,
    expected: dict[str, Any],
) -> list[CriterionResult]:
    exp_consensus = bool(expected["expected_consensus_passed"])
    exp_min_score = int(expected["expected_min_score"])
    exp_industry = bool(expected["expected_is_target_industry"])

    actual_consensus = output.passed_consensus
    actual_score = output.lead_score
    actual_industry = _actual_is_target_industry(output)

    criteria: list[CriterionResult] = []

    criteria.append(
        CriterionResult(
            name="consensus_passed",
            expected=exp_consensus,
            actual=actual_consensus,
            passed=actual_consensus == exp_consensus,
        )
    )

    score_ok = actual_score >= exp_min_score
    criteria.append(
        CriterionResult(
            name="min_lead_score",
            expected=f">= {exp_min_score}",
            actual=actual_score,
            passed=score_ok,
            detail=f"score={actual_score}",
        )
    )

    if actual_industry is None:
        industry_ok = not exp_industry
        industry_detail = "website_signals unavailable (treated as non-target)"
    else:
        industry_ok = actual_industry == exp_industry
        industry_detail = ""

    criteria.append(
        CriterionResult(
            name="is_target_industry",
            expected=exp_industry,
            actual=actual_industry,
            passed=industry_ok,
            detail=industry_detail,
        )
    )

    return criteria


def _classify_outcome(
    exp_consensus: bool, case_passed: bool, actual_consensus: bool
) -> str:
    """TP / TN / FP / FN relative to consensus expectation."""
    if exp_consensus and case_passed:
        return "TP"
    if not exp_consensus and case_passed:
        return "TN"
    if not exp_consensus and actual_consensus:
        return "FP"
    if exp_consensus and not actual_consensus:
        return "FN"
    return "MISS"  # criteria failed but consensus direction ambiguous


async def _run_single_case(case: dict[str, Any]) -> CaseResult:
    case_id = case.get("case_id", case["inputs"]["business_name"])
    description = case.get("description", "")
    inputs = case["inputs"]
    expected = case["expected_outputs"]

    t0 = time.perf_counter()
    try:
        with patch(
            "src.agent.nodes.enrichment.get_hunter_service",
            new=_mock_get_hunter_service,
        ):
            output = await run_sourcing_agent(
                query=inputs["query"],
                business_name=inputs["business_name"],
                location=inputs["location"],
                website_url=inputs.get("website_url"),
                investment_thesis=inputs.get("investment_thesis"),
            )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return CaseResult(
            case_id=case_id,
            description=description,
            business_name=inputs["business_name"],
            passed=False,
            execution_time_seconds=elapsed,
            errors=[str(exc)],
        )

    elapsed = time.perf_counter() - t0
    criteria = _evaluate_case_output(output, expected)
    case_passed = all(c.passed for c in criteria)

    return CaseResult(
        case_id=case_id,
        description=description,
        business_name=inputs["business_name"],
        passed=case_passed,
        criteria=criteria,
        lead_score=output.lead_score,
        execution_time_seconds=elapsed,
        errors=list(output.errors_encountered or []),
    )


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset must be a non-empty JSON array: {path}")
    return data


def _print_case_result(result: CaseResult, index: int, total: int) -> None:
    status = _c("PASS", GREEN + BOLD) if result.passed else _c("FAIL", RED + BOLD)
    print(f"\n{_hr()}")
    print(
        f"{_c(f'[{index}/{total}]', CYAN)} {status}  "
        f"{_c(result.case_id, BOLD)} — {result.business_name}"
    )
    if result.description:
        print(f"{_c('Description:', DIM)} {result.description}")
    if result.execution_time_seconds is not None:
        print(f"{_c('Runtime:', DIM)} {result.execution_time_seconds:.1f}s  "
              f"{_c('Score:', DIM)} {result.lead_score}/100")

    for crit in result.criteria:
        icon = _c("✓", GREEN) if crit.passed else _c("✗", RED)
        detail = f" ({crit.detail})" if crit.detail else ""
        print(
            f"  {icon} {crit.name}: expected={crit.expected!r}, "
            f"actual={crit.actual!r}{detail}"
        )

    if result.errors:
        print(f"  {_c('Pipeline errors:', YELLOW)}")
        for err in result.errors[:3]:
            print(f"    • {err[:120]}{'...' if len(err) > 120 else ''}")


def _print_summary(
    results: list[CaseResult],
    cases: list[dict[str, Any]],
    dataset_path: Path,
    total_wall_seconds: float,
) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    accuracy = (passed / total * 100.0) if total else 0.0

    tp = tn = fp = fn = 0
    for result, case in zip(results, cases):
        exp = bool(case["expected_outputs"]["expected_consensus_passed"])
        actual_consensus = next(
            (c.actual for c in result.criteria if c.name == "consensus_passed"),
            False,
        )
        label = _classify_outcome(exp, result.passed, bool(actual_consensus))
        if label == "TP":
            tp += 1
        elif label == "TN":
            tn += 1
        elif label == "FP":
            fp += 1
        elif label == "FN":
            fn += 1

    print(f"\n{_hr('═')}")
    print(_c("  GOLDEN DATASET EVALUATION SUMMARY", BOLD + MAGENTA))
    print(_hr("═"))
    print(f"  Dataset     : {dataset_path}")
    print(f"  Cases       : {total}")
    print(f"  Wall time   : {total_wall_seconds:.1f}s")
    print(f"  Hunter.io   : {_c('MOCKED (Node 4)', YELLOW)}")
    print()
    print(f"  {_c('Overall', BOLD)}")
    print(f"    Passed    : {_c(str(passed), GREEN)}")
    print(f"    Failed    : {_c(str(failed), RED if failed else DIM)}")
    print(f"    Accuracy  : {_c(f'{accuracy:.1f}%', GREEN if accuracy >= 80 else YELLOW)}")
    print()
    print(f"  {_c('Consensus classification (all criteria must pass for TN/TP)', BOLD)}")
    print(f"    True Positives  : {tp}")
    print(f"    True Negatives  : {tn}")
    print(f"    False Positives : {_c(str(fp), RED if fp else DIM)}")
    print(f"    False Negatives : {_c(str(fn), RED if fn else DIM)}")
    print(_hr("═"))

    if failed:
        failed_ids = [r.case_id for r in results if not r.passed]
        print(f"\n{_c('Failed cases:', RED)} {', '.join(failed_ids)}")
        sys.exit(1)


async def run_evaluation(dataset_path: Path) -> None:
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("httpx", "httpcore", "crawl4ai", "playwright"):
        logging.getLogger(name).setLevel(logging.WARNING)

    cases = _load_dataset(dataset_path)
    print(_c("\nLead-Gen Agent — Golden Dataset Evaluation", BOLD + CYAN))
    print(f"Loading {len(cases)} cases from {dataset_path}")
    print(_c("Node 4 (Hunter.io) patched via unittest.mock — no API credits used.\n", DIM))

    wall_start = time.perf_counter()
    results: list[CaseResult] = []

    for i, case in enumerate(cases, start=1):
        print(_c(f"Running case {i}/{len(cases)}: {case.get('case_id', '?')}...", DIM))
        result = await _run_single_case(case)
        results.append(result)
        _print_case_result(result, i, len(cases))

    _print_summary(results, cases, dataset_path, time.perf_counter() - wall_start)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate run_sourcing_agent against tests/eval_dataset.json"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to eval JSON (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.dataset.resolve()))


if __name__ == "__main__":
    main()
