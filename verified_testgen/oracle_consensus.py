"""Accept expected output only when independent oracles agree."""
# 多个Oracle生成相同结果才判断为正确
import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .problem_adapter import normalize_stdout


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    stdout: str | None
    error: str | None = None


@dataclass(frozen=True)
class OracleCandidate:
    code: str
    raw_response: str
    algorithm_label: str


@dataclass(frozen=True)
class ConsensusResult:
    expected: str | None
    supporting_oracles: tuple[int, ...]
    results: tuple[ExecutionResult, ...]


AsyncExecutor = Callable[[str, str], Awaitable[ExecutionResult]]


async def consensus_for_input(
    oracles: list[OracleCandidate], test_input: str, executor: AsyncExecutor, minimum_support: int = 2
) -> ConsensusResult:
    #并发所有Oracle
    results = tuple(await asyncio.gather(*(executor(oracle.code, test_input) for oracle in oracles)))
    groups: dict[str, list[int]] = defaultdict(list)
    for index, result in enumerate(results):
        if result.status == "accepted" and result.stdout is not None:
            groups[normalize_stdout(result.stdout)].append(index)
    if not groups:
        return ConsensusResult(None, (), results)
    expected, supporters = max(groups.items(), key=lambda item: (len(item[1]), item[0]))
    if len(supporters) < minimum_support:
        return ConsensusResult(None, (), results)
    return ConsensusResult(expected, tuple(supporters), results)
