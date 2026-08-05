import pytest

from verified_testgen.input_generation import CandidateInput, filter_candidates, limit_candidates, normalize_and_deduplicate


def test_normalization_and_filtering_are_deterministic():
    candidates = [CandidateInput("1", "base", "a"), CandidateInput("1\n", "corner", "b")]
    assert len(normalize_and_deduplicate(candidates)) == 1
    assert filter_candidates(candidates, lambda value: value == "1\n", 10)[0].category == "base"


def test_limit_candidates_deduplicates_before_truncating():
    candidates = [CandidateInput("1", "base", "a"), CandidateInput("1\n", "duplicate", "b"), CandidateInput("2\n", "edge", "c")]
    assert [case.input for case in limit_candidates(candidates, 2)] == ["1\n", "2\n"]
    with pytest.raises(ValueError, match="positive"):
        limit_candidates(candidates, 0)
