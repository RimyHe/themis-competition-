import json

import pytest

from verified_testgen.io import load_problems, submission_record
from verified_testgen.problem_adapter import CompetitionProblem, SelectedTestCase


def test_submission_is_limited_and_uses_original_solutions():
    problem = CompetitionProblem("42", "question", ("print(1)",))
    case = SelectedTestCase("1\n", "1\n")
    assert submission_record(problem, [case])["solutions"] == ["print(1)"]
    with pytest.raises(ValueError, match="maximum"):
        submission_record(problem, [case] * 6)


def test_load_problems_rejects_duplicate_ids(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{"id": "1", "question": "q", "solutions": []}] * 2))
    with pytest.raises(ValueError, match="duplicate"):
        load_problems(source)
