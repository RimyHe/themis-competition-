import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from verified_testgen.constraints import ThemisInputPipeline, balanced_themis_candidates
from verified_testgen.input_generation import CandidateInput


CONSTRAINT = """```python
class ProblemConstraint:
    def parse_input(self, raw_input):
        return {\"value\": raw_input}

    def validate(self, raw_input):
        return True
```"""

TESTGEN = """```python
class TestGenerator:
    def generate_base_case(self):
        return []

    def generate_corner_case(self):
        return []

    def generate_complex_case(self):
        return []
```"""


class FakeGenerator:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def generate_sample_trace(self, prompt, task_id):
        return {"task_id": task_id, "response_text": next(self.responses), "finish_reason": "stop", "error": None}


@dataclass
class FakeStats:
    generated: dict[str, int]
    filtered: dict[str, int]


def test_themis_pipeline_uses_validated_categories(monkeypatch):
    def fake_validate(constraint_code, testgen_code, timeout):
        assert "ProblemConstraint" in constraint_code
        assert "TestGenerator" in testgen_code
        return SimpleNamespace(
            testcases={"base": ["1", "1\n"], "corner": ["0\n"], "complex": ["2\n"]},
            stats=FakeStats(generated={"base": 2, "corner": 1, "complex": 1}, filtered={"base": 2, "corner": 1, "complex": 1}),
        )

    monkeypatch.setattr("verified_testgen.constraints._validate_single_problem_isolated", fake_validate)
    candidates, artifacts = asyncio.run(ThemisInputPipeline(FakeGenerator([CONSTRAINT, TESTGEN])).generate_candidates("question"))

    assert artifacts.error is None
    assert [(item.input, item.category) for item in candidates] == [("1\n", "base"), ("0\n", "corner"), ("2\n", "complex")]
    assert artifacts.validation_stats["filtered"]["complex"] == 1


def test_themis_pipeline_rejects_incomplete_generated_component():
    incomplete = "```python\nclass ProblemConstraint:\n    def validate(self, raw_input):\n        return True\n```"
    candidates, artifacts = asyncio.run(ThemisInputPipeline(FakeGenerator([incomplete, TESTGEN])).generate_candidates("question"))
    assert candidates == []
    assert "parse_input" in artifacts.error


def test_balanced_themis_candidates_preserves_categories_under_a_small_limit():
    candidates = [
        CandidateInput("a\n", "base", "themis-base"),
        CandidateInput("b\n", "base", "themis-base"),
        CandidateInput("c\n", "corner", "themis-corner"),
        CandidateInput("d\n", "complex", "themis-complex"),
    ]
    assert [item.category for item in balanced_themis_candidates(candidates, 3)] == ["base", "corner", "complex"]
