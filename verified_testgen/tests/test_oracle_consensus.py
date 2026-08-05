import asyncio

from verified_testgen.oracle_consensus import ExecutionResult, OracleCandidate, consensus_for_input


async def fake_executor(code: str, test_input: str) -> ExecutionResult:
    return ExecutionResult("accepted", "7" if code != "wrong" else "8")


def test_two_matching_oracles_produce_consensus():
    oracles = [
        OracleCandidate("one", "", "brute"),
        OracleCandidate("two", "", "simulation"),
        OracleCandidate("wrong", "", "other"),
    ]
    result = asyncio.run(consensus_for_input(oracles, "1\n", fake_executor))
    assert result.expected == "7\n"
    assert result.supporting_oracles == (0, 1)
