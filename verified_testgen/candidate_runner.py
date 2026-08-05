"""Remote mini-judge client and differential execution matrix construction."""

import asyncio
from dataclasses import dataclass

import aiohttp

from .oracle_consensus import ExecutionResult
from .problem_adapter import CompetitionProblem, SelectedTestCase, normalize_stdout


@dataclass(frozen=True)
class JudgeConfig:
    url: str
    timeout_seconds: int = 120
    time_limit_seconds: int = 5
    memory_limit_mb: int = 1024
    max_concurrency: int = 1


class MiniJudgeClient:
    def __init__(self, config: JudgeConfig):
        self.config = config

    async def run(self, code: str, test_input: str) -> ExecutionResult:
        payload = {
            "code": code,
            "language": "python",
            "mode": "execution",
            "test_cases": [{"input": test_input, "expected": ""}],
            "time_limit": self.config.time_limit_seconds,
            "memory_limit": self.config.memory_limit_mb,
            "security_check": True,
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.config.url, json=payload) as response:
                    response.raise_for_status()
                    body = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return ExecutionResult("infrastructure_error", None, str(exc))
        cases = body.get("test_case_results")
        if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
            return ExecutionResult("infrastructure_error", None, "malformed judge response")
        case = cases[0]
        return ExecutionResult(str(case.get("status", "infrastructure_error")), case.get("actual_output"), case.get("error_message"))


async def build_detection_matrix(  #输入问题，测试，judge客户端，限制并发
    problem: CompetitionProblem,
    labeled_cases: list[SelectedTestCase],
    client: MiniJudgeClient,
    semaphore: asyncio.Semaphore | None = None,
) -> list[SelectedTestCase]:
    semaphore = semaphore or asyncio.Semaphore(client.config.max_concurrency)

    async def run_solution(code: str, test_input: str) -> ExecutionResult:
        async with semaphore:
            return await client.run(code, test_input)

    enriched = []
    for case in labeled_cases:
        results = await asyncio.gather(*(run_solution(code, case.input) for code in problem.solutions))
        expected = normalize_stdout(case.expected)
        detected = {
            index
            for index, result in enumerate(results)
            if result.status not in {"accepted", "infrastructure_error"}
            or (result.stdout is not None and normalize_stdout(result.stdout) != expected)
        }
        enriched.append(
            SelectedTestCase(case.input, expected, case.category, frozenset(detected))
        )
    return enriched
