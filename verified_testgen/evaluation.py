"""Development-set metrics for oracle reliability and local detection coverage."""

from dataclasses import dataclass

from .problem_adapter import SelectedTestCase, normalize_stdout


@dataclass(frozen=True)
class CoverageMetrics:
    detected: int
    candidate_solutions: int
    local_coverage: float


def coverage_metrics(cases: list[SelectedTestCase], solution_count: int) -> CoverageMetrics:
    detected = len(frozenset().union(*(case.detected_solutions for case in cases))) if cases else 0
    return CoverageMetrics(detected, solution_count, detected / solution_count if solution_count else 0.0)


def oracle_matches_expected(oracle_output: str, official_expected: str) -> bool:
    return normalize_stdout(oracle_output) == normalize_stdout(official_expected)


def edr(selected: list[SelectedTestCase], known_error_indices: set[int]) -> float:
    if not known_error_indices:
        return 0.0
    covered = frozenset().union(*(case.detected_solutions for case in selected)) if selected else frozenset()
    return len(covered & known_error_indices) / len(known_error_indices)
