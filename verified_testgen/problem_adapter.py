"""Typed boundary between the competition JSON and the pipeline."""
#对格式的处理
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompetitionProblem:
    problem_id: str
    question: str
    solutions: tuple[str, ...]


@dataclass(frozen=True)
class SelectedTestCase:
    input: str
    expected: str
    category: str = "unknown"
    detected_solutions: frozenset[int] = field(default_factory=frozenset)


def problem_from_record(record: dict[str, Any]) -> CompetitionProblem:
    problem_id = str(record.get("id", ""))
    question = record.get("question")
    solutions = record.get("solutions")
    if not problem_id:
        raise ValueError("problem id is required")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{problem_id}: question must be a non-empty string")
    if not isinstance(solutions, list) or not all(isinstance(code, str) for code in solutions):
        raise ValueError(f"{problem_id}: solutions must be a list of strings")
    return CompetitionProblem(problem_id=problem_id, question=question, solutions=tuple(solutions))


def test_case_from_record(record: dict[str, Any]) -> SelectedTestCase:
    test_input = record.get("input")
    expected = record.get("expected")
    if not isinstance(test_input, str) or not isinstance(expected, str):
        raise ValueError("test case requires string input and expected fields")
    detected = record.get("detected_solutions", [])
    if not isinstance(detected, list) or not all(isinstance(index, int) for index in detected):
        raise ValueError("detected_solutions must be a list of integers")
    return SelectedTestCase(
        input=normalize_stdin(test_input),
        expected=normalize_stdout(expected),
        category=str(record.get("category", "unknown")),
        detected_solutions=frozenset(detected),
    )


def normalize_stdin(value: str) -> str:
    return value.rstrip("\n") + "\n"


def normalize_stdout(value: str) -> str:
    return value.rstrip() + "\n"
