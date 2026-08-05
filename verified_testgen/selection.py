"""Maximum-coverage selection under the competition's five-test budget."""

from itertools import combinations

from .problem_adapter import SelectedTestCase

MAX_SELECTED = 5


def _score(cases: tuple[SelectedTestCase, ...]) -> tuple[int, int, int, tuple[str, ...]]:
    covered = frozenset().union(*(case.detected_solutions for case in cases)) if cases else frozenset()
    categories = len({case.category for case in cases})
    total_length = sum(len(case.input) for case in cases)
    return (len(covered), categories, -total_length, tuple(case.input for case in cases))


def greedy_select(candidates: list[SelectedTestCase], limit: int = MAX_SELECTED) -> list[SelectedTestCase]:
    remaining = list(candidates)
    selected = []
    covered = frozenset()
    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda case: (len(case.detected_solutions - covered), -len(case.input), case.input),
        )
        if not best.detected_solutions - covered:
            break
        selected.append(best)
        covered |= best.detected_solutions
        remaining.remove(best)
    return selected


def select_cases(candidates: list[SelectedTestCase], limit: int = MAX_SELECTED) -> list[SelectedTestCase]:
    """Use exact search for small pools and deterministic greedy selection otherwise."""
    if limit < 1 or limit > MAX_SELECTED:
        raise ValueError(f"limit must be between 1 and {MAX_SELECTED}")
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) > 20:
        return greedy_select(candidates, limit)
    best: tuple[SelectedTestCase, ...] = ()
    for size in range(1, min(limit, len(candidates)) + 1):
        for subset in combinations(candidates, size):
            if _score(subset) > _score(best):
                best = subset
    return list(best) if _score(best)[0] else []
