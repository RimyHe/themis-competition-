"""Competition input/output handling with strict submission validation."""
#规范输出内容的格式
import json
from pathlib import Path
from typing import Any

from .problem_adapter import CompetitionProblem, SelectedTestCase, problem_from_record

MAX_TEST_CASES = 5


def load_problems(path: Path) -> list[CompetitionProblem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array")
    problems = [problem_from_record(record) for record in raw]
    ids = [problem.problem_id for problem in problems]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate problem ids")
    return problems


def submission_record(problem: CompetitionProblem, test_cases: list[SelectedTestCase]) -> dict[str, Any]:
    if len(test_cases) > MAX_TEST_CASES:
        raise ValueError(f"{problem.problem_id}: submission has {len(test_cases)} tests; maximum is {MAX_TEST_CASES}")
    cases = []
    for case in test_cases:
        if not case.input.endswith("\n") or not case.expected.endswith("\n"):
            raise ValueError(f"{problem.problem_id}: input and expected must end with a newline")
        cases.append({"input": case.input, "expected": case.expected})
    return {"problem_id": problem.problem_id, "solutions": list(problem.solutions), "test_cases": cases}


def write_submission(output_dir: Path, problem: CompetitionProblem, test_cases: list[SelectedTestCase]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = submission_record(problem, test_cases)
    destination = output_dir / f"test_cases_{problem.problem_id}.jsonl"
    destination.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return destination


def load_prepared_cases(path: Path) -> dict[str, list[SelectedTestCase]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object keyed by problem id")
    from .problem_adapter import test_case_from_record

    result = {}
    for problem_id, cases in raw.items():
        if not isinstance(cases, list):
            raise ValueError(f"{problem_id}: expected a list of cases")
        result[str(problem_id)] = [test_case_from_record(case) for case in cases]
    return result
