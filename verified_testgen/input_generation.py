"""Stage 2/3 candidate-pool management without executing generated code locally."""

import random
from collections.abc import Callable
from dataclasses import dataclass

from .problem_adapter import normalize_stdin


@dataclass(frozen=True)
class CandidateInput:
    input: str
    category: str
    source: str


def limit_candidates(candidates: list[CandidateInput], limit: int) -> list[CandidateInput]:
    """Keep a deterministic bounded pool after normalizing and deduplicating."""
    if limit < 1:
        raise ValueError("limit must be positive")
    return normalize_and_deduplicate(candidates)[:limit]


def normalize_and_deduplicate(candidates: list[CandidateInput]) -> list[CandidateInput]:
    seen = set()
    result = []
    for candidate in candidates:
        normalized = normalize_stdin(candidate.input)
        if normalized not in seen:
            seen.add(normalized)
            result.append(CandidateInput(normalized, candidate.category, candidate.source))
    return result


def filter_candidates(
    candidates: list[CandidateInput], validator: Callable[[str], bool], max_input_bytes: int
) -> list[CandidateInput]:
    return [
        candidate
        for candidate in normalize_and_deduplicate(candidates)
        if len(candidate.input.encode("utf-8")) <= max_input_bytes and validator(candidate.input)
    ]


def interleave_categories(candidates: list[CandidateInput], limit: int) -> list[CandidateInput]:
    """Deterministically cap candidates without allowing one category to starve others."""
    if limit < 1:
        raise ValueError("limit must be positive")
    groups: dict[str, list[CandidateInput]] = {}
    for candidate in normalize_and_deduplicate(candidates):
        groups.setdefault(candidate.category, []).append(candidate)
    selected = []
    while groups and len(selected) < limit:
        for category in tuple(groups):
            selected.append(groups[category].pop(0))
            if not groups[category]:
                del groups[category]
            if len(selected) == limit:
                break
    return selected


def mutate_lines(seed_input: str, seed: int, attempts: int = 8) -> list[CandidateInput]:
    """Deterministic lightweight mutations for a later validator/oracle pass."""
    rng = random.Random(seed)
    lines = seed_input.rstrip("\n").split("\n")
    result = []
    for index in range(attempts):
        mutated = list(lines)
        if mutated:
            line_index = rng.randrange(len(mutated))
            tokens = mutated[line_index].split()
            if len(tokens) > 1:
                rng.shuffle(tokens)
                mutated[line_index] = " ".join(tokens)
            elif tokens and tokens[0].lstrip("-").isdigit():
                delta = rng.choice([-1, 1])
                mutated[line_index] = str(int(tokens[0]) + delta)
        result.append(CandidateInput("\n".join(mutated) + "\n", "differential", f"line-mutation:{index}"))
    return result
