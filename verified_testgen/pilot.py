"""Constraint-aware competition pilot using Themis, local Qwen, and mini-judge."""
#核心，给定一道题，生成候选测试输入和多个参考解（oracle），通过共识确认正确输出，再找出最多能检测错误候选程序的 5 个测试。
# 恢复模式：1. Oracle有效的少于2个 再生成 2. 输入有效的少于6个 再调用
import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from themis.core.completion.generator import LLMGenerator
from themis.core.completion.utils import GenerationConfig

from .candidate_runner import JudgeConfig, MiniJudgeClient, build_detection_matrix
from .constraints import ThemisInputPipeline, balanced_themis_candidates
from .io import load_problems, write_submission
from .oracle_consensus import OracleCandidate, consensus_for_input
from .oracle_generation import static_check_oracle
from .problem_adapter import SelectedTestCase
from .selection import select_cases


ORACLE_APPROACHES = ( #对同一题，写三个不同思路的参考程序
    "direct brute-force enumeration or exhaustive simulation", #暴力枚举/穷举模拟
    "a simple dynamic program or recurrence with independently derived states", #简单动态规划/递推关系
    "a literal specification-following simulation with clear invariants", #严格遵循规范的字面仿真实现
)
RECOVERY_ORACLE_APPROACHES = ( #兜底策略，当前三个 oracle 中有效的不到两个时，再要求生成偏“最小数据范围暴力搜索”和“按题意独立实现”的版本。
    "a table-driven exhaustive search over the smallest legal domain",
    "a separate reference implementation derived from the input/output specification",
)
MAX_CANDIDATE_INPUTS = 12 #每题最多12个候选输入


def _oracle_prompt(question: str, approach: str) -> str: #生成Oracle的提示词 
    return f"""Write one self-contained Python ACM stdin/stdout oracle for the problem below.
It is used only for deliberately small inputs. Use {approach}; do not optimize.
It must be deterministic and mathematically correct. Do not use randomness.
Do not read files, access the network, create processes, or include an explanation.
Return exactly one fenced Python code block.

Problem:
{question}"""


def _oracle_from_trace(trace: dict, approach: str) -> OracleCandidate | None: #将模型相应转换成可执行Oracle
    from themis.utils.sanitize import extract_code_block

    response = trace.get("response_text") or ""
    try:
        code = extract_code_block(response)
        static_check_oracle(code)
    except (SyntaxError, ValueError) as exc:
        trace["static_check_error"] = str(exc)
        return None
    return OracleCandidate(code, response, approach)


async def _generate(generator: LLMGenerator, prompt: str, task_id: str) -> dict:
    return await generator.generate_sample_trace(prompt, task_id=task_id)


async def run_problem(
    problem,
    generator: LLMGenerator,
    client: MiniJudgeClient,
    run_dir: Path,
    judge_semaphore: asyncio.Semaphore | None = None,
) -> dict:
    problem_dir = run_dir / problem.problem_id #创建目录
    problem_dir.mkdir(parents=True, exist_ok=True)

    themis = ThemisInputPipeline(generator)
    candidates, themis_artifacts = await themis.generate_candidates(problem.question)
    candidates = balanced_themis_candidates(candidates, MAX_CANDIDATE_INPUTS)

    oracle_traces = []
    oracles = []
    for index, approach in enumerate(ORACLE_APPROACHES): #生成三份Oracle
        trace = await _generate(generator, _oracle_prompt(problem.question, approach), f"oracle-{index}")
        oracle_traces.append(trace)
        oracle = _oracle_from_trace(trace, approach)
        if oracle is not None:
            oracles.append(oracle)

    # A malformed response must not make the entire problem permanently untestable.
    if len(oracles) < 2:
        for index, approach in enumerate(RECOVERY_ORACLE_APPROACHES):
            trace = await _generate(generator, _oracle_prompt(problem.question, approach), f"oracle-recovery-{index}")
            oracle_traces.append(trace)
            oracle = _oracle_from_trace(trace, approach)
            if oracle is not None:
                oracles.append(oracle)
            if len(oracles) >= 2:
                break

    semaphore = judge_semaphore or asyncio.Semaphore(1)

    async def execute(code: str, test_input: str): #局部执行器
        async with semaphore:
            return await client.run(code, test_input)

    async def label_candidates() -> tuple[list[SelectedTestCase], list[dict]]:
        labelled = []
        consensus_records = []
        for candidate in candidates:
            consensus = await consensus_for_input(oracles, candidate.input, execute)
            consensus_records.append(
                {
                    "input": candidate.input,
                    "category": candidate.category,
                    "expected": consensus.expected,
                    "supporting_oracles": list(consensus.supporting_oracles),
                    "results": [asdict(result) for result in consensus.results],
                }
            )
            if consensus.expected is not None:
                labelled.append(SelectedTestCase(candidate.input, consensus.expected, candidate.category))
        return labelled, consensus_records

    labelled, consensus_records = await label_candidates() if len(oracles) >= 2 else ([], [])

    # Keep the two-oracle acceptance rule, but give disagreeing initial references
    # two genuinely different formulations before abandoning the problem.
    if not labelled and candidates:
        for index, approach in enumerate(RECOVERY_ORACLE_APPROACHES):
            trace = await _generate(generator, _oracle_prompt(problem.question, approach), f"oracle-consensus-recovery-{index}")
            oracle_traces.append(trace)
            oracle = _oracle_from_trace(trace, approach)
            if oracle is not None:
                oracles.append(oracle)
        labelled, retry_records = await label_candidates() if len(oracles) >= 2 else ([], [])
        consensus_records.extend(retry_records)

    enriched = await build_detection_matrix(problem, labelled, client, semaphore)
    selected = select_cases(enriched)
    output_path = write_submission(problem_dir / "output", problem, selected)
    (problem_dir / "oracle_traces.json").write_text(json.dumps(oracle_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    (problem_dir / "themis.json").write_text(
        json.dumps(themis_artifacts.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (problem_dir / "consensus.json").write_text(json.dumps(consensus_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "problem_id": problem.problem_id,
        "oracle_candidates": len(oracles),
        "candidate_inputs": len(candidates),
        "themis_error": themis_artifacts.error,
        "consensus_cases": len(labelled),
        "selected_cases": len(selected),
        "detected_solutions": sorted(set().union(*(case.detected_solutions for case in selected))) if selected else [],
        "output": str(output_path),
    }


async def run(args) -> None:
    problems = load_problems(args.input)
    if args.problem_id:
        problems = [problem for problem in problems if problem.problem_id == args.problem_id]
    elif args.limit:
        problems = problems[: args.limit]
    if not problems:
        raise ValueError("no requested problems found")
    run_dir = args.output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True)
    config = GenerationConfig(
        base_url=args.model_url,
        model=args.model,
        temperature=0,
        top_p=1,
        max_tokens=args.max_tokens,
        timeout=args.model_timeout,
    )
    client = MiniJudgeClient(JudgeConfig(url=args.judge_url, max_concurrency=1))
    generator = LLMGenerator(config)
    summary = [await run_problem(problem, generator, client, run_dir) for problem in problems]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(args.input),
        "model": config.model,
        "model_url": config.base_url,
        "judge_url": args.judge_url,
        "model_server_gpu": args.model_server_gpu,
        "strategy": "S1 pilot: Themis constraint/testgen validation, two-oracle consensus, maximum coverage",
        "results": summary,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a conservative, auditable competition pilot.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--model-timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8000/api/v1/judge")
    parser.add_argument(
        "--model-server-gpu",
        required=True,
        help="Physical GPU ID(s) used by the separately launched model server; recorded for reproducibility.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
