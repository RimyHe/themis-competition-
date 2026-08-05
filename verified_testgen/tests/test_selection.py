from verified_testgen.problem_adapter import SelectedTestCase
from verified_testgen.selection import select_cases


def test_exact_selection_beats_a_shorter_overlap_case():
    cases = [
        SelectedTestCase("a\n", "x\n", detected_solutions=frozenset({0, 1})),
        SelectedTestCase("b\n", "x\n", detected_solutions=frozenset({2})),
        SelectedTestCase("c\n", "x\n", detected_solutions=frozenset({3})),
    ]
    selected = select_cases(cases, limit=2)
    assert len(selected) == 2
    assert {0, 1}.issubset(set().union(*(case.detected_solutions for case in selected)))


def test_selection_omits_zero_gain_cases():
    assert select_cases([SelectedTestCase("a\n", "x\n")]) == []
