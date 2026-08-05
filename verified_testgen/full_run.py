"""Resumable full-dataset competition runner for the local GPU 7 service."""
#用于对整个题目数据集批量生成测试

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from themis.core.completion.generator import LLMGenerator
from themis.core.completion.utils import GenerationConfig

from .candidate_runner import JudgeConfig, MiniJudgeClient
from .io import load_problems, write_submission
from .pilot import run_problem


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _completed(problem_dir: Path) -> bool: #有执行摘要（summary）和output（正式输出）
    summary_path = problem_dir / "summary.json"
    if not summary_path.is_file() or not any((problem_dir / "output").glob("*.jsonl")):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return not summary.get("error") and not summary.get("themis_error")


def _failure_record(problem, kind: str, reason: str, problem_dir: Path) -> dict:
    return {
        "problem_id": problem.problem_id,
        "kind": kind,
        "reason": reason,
        "recorded_at": datetime.now(UTC).isoformat(),
        "input": {
            "id": problem.problem_id,
            "question": problem.question,
            "solutions": list(problem.solutions),
        },
        "summary": str(problem_dir / "summary.json"),
        "themis_artifacts": str(problem_dir / "themis.json"),
    }


def _record_failure(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _backfill_failures(problems, run_dir: Path, failure_path: Path) -> None:
    """Preserve failure reasons from completed attempts before resume retries them."""
    existing = set()
    if failure_path.is_file():
        for line in failure_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                existing.add((record.get("problem_id"), record.get("kind"), record.get("reason")))
            except json.JSONDecodeError:
                continue
    for problem in problems:
        summary_path = run_dir / problem.problem_id / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if summary.get("error"):
            kind, reason = "pipeline_exception", str(summary["error"])
        elif summary.get("themis_error"):
            kind, reason = "themis_generation_failure", str(summary["themis_error"])
        else:
            continue
        if (problem.problem_id, kind, reason) not in existing:
            _record_failure(failure_path, _failure_record(problem, kind, reason, run_dir / problem.problem_id))


async def run(args: argparse.Namespace) -> None:
    problems = load_problems(args.input)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")
    scoped_problems = [
        problem for index, problem in enumerate(problems) if index % args.shard_count == args.shard_index
    ]
    if not 0 <= args.subshard_index < args.subshard_count:
        raise ValueError("--subshard-index must be in [0, --subshard-count)")
    assigned_problems = [
        problem
        for index, problem in enumerate(scoped_problems)
        if index % args.subshard_count == args.subshard_index
    ]
    #如果指定了旧目录就使用，从而支持中断后续跑
    run_dir = args.run_dir or args.output_dir / datetime.now(UTC).strftime("gpu7-s1-%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_path = run_dir / "failures.jsonl"
    if args.subshard_count > 1:
        failure_path = run_dir / f"failures-subshard-{args.subshard_index}-of-{args.subshard_count}.jsonl"
    _backfill_failures(assigned_problems, run_dir, failure_path)
    config = GenerationConfig(
        base_url=args.model_url,
        model=args.model,
        temperature=0,
        top_p=1,
        max_tokens=args.max_tokens,
        timeout=args.model_timeout,
    )
    generator = LLMGenerator(config) #调模型，生成Oracle和测试输入
    client = MiniJudgeClient(JudgeConfig(url=args.judge_url, max_concurrency=1)) #请求mini-judge，不允许并行
    judge_semaphore = asyncio.Semaphore(1)
    work = [problem for problem in assigned_problems if not _completed(run_dir / problem.problem_id)]
    progress_path = (
        run_dir / f"progress-subshard-{args.subshard_index}-of-{args.subshard_count}.json"
        if args.subshard_count > 1
        else run_dir / "progress.json"
    )
    progress = {
        "total": len(assigned_problems),
        "completed": len(assigned_problems) - len(work),
        "failed": 0,
        "updated_at": None,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "subshard_index": args.subshard_index,
        "subshard_count": args.subshard_count,
    }
    lock = asyncio.Lock()

    async def process(problem) -> None: #一题的任务生命周期
        problem_dir = run_dir / problem.problem_id
        try:
            #用 run_problem 生成与筛选测试
            result = await asyncio.wait_for(
                run_problem(problem, generator, client, run_dir, judge_semaphore),
                timeout=args.problem_timeout,
            )
            _write_json(problem_dir / "summary.json", result)
            if result.get("themis_error"):
                async with lock:
                    _record_failure(
                        failure_path,
                        _failure_record(
                            problem, "themis_generation_failure", result["themis_error"], problem_dir
                        ),
                    )
        except Exception as exc:
            problem_dir.mkdir(parents=True, exist_ok=True)
            output_path = write_submission(problem_dir / "output", problem, [])
            result = {"problem_id": problem.problem_id, "selected_cases": 0, "output": str(output_path), "error": str(exc)}
            _write_json(problem_dir / "summary.json", result)
            async with lock:
                progress["failed"] += 1
                _record_failure(failure_path, _failure_record(problem, "pipeline_exception", str(exc), problem_dir))
        finally:
            async with lock:
                progress["completed"] += 1
                progress["updated_at"] = datetime.now(UTC).isoformat()
                _write_json(progress_path, progress)

    queue: asyncio.Queue = asyncio.Queue() #并发实现
    for problem in work:
        queue.put_nowait(problem)

    async def worker() -> None:
        while not queue.empty():
            problem = await queue.get()
            try:
                await process(problem)
            finally:
                queue.task_done()

    await asyncio.gather(*(worker() for _ in range(args.model_workers)))

    aggregate = None
    if args.shard_count == 1:
        records = []
        for problem in problems:
            file_path = run_dir / problem.problem_id / "output" / f"test_cases_{problem.problem_id}.jsonl"
            records.append(json.loads(file_path.read_text(encoding="utf-8")))
        aggregate = run_dir / "submission.jsonl"
        aggregate.write_text(
            "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "input": str(args.input),
        "model": config.model,
        "model_url": config.base_url,
        "judge_url": args.judge_url,
        "model_server_gpu": args.model_server_gpu,
        "model_workers": args.model_workers,
        "strategy": "S1: Themis constraint/testgen validation, two-oracle consensus, exact <=5 coverage",
        "submission": str(aggregate) if aggregate else None,
        "problems": len(assigned_problems),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "subshard_index": args.subshard_index,
        "subshard_count": args.subshard_count,
    }
    manifest_path = (
        run_dir / f"manifest-subshard-{args.subshard_index}-of-{args.subshard_count}.json"
        if args.subshard_count > 1
        else run_dir / "manifest.json"
    )
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all competition problems with resumable GPU 7 generation.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--model-timeout", type=int, default=300)
    parser.add_argument(
        "--problem-timeout",
        type=int,
        default=600,
        help="Maximum wall-clock seconds for one problem before recording a recoverable failure.",
    )
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8000/api/v1/judge")
    parser.add_argument("--model-workers", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--subshard-index",
        type=int,
        default=0,
        help="Optional disjoint partition within the selected shard.",
    )
    parser.add_argument(
        "--subshard-count",
        type=int,
        default=1,
        help="Number of disjoint partitions within the selected shard.",
    )
    parser.add_argument(
        "--model-server-gpu",
        required=True,
        help="Physical GPU ID(s) used by the separately launched model server; recorded for reproducibility.",
    )
    args = parser.parse_args()
    #接收题目json，输出目录，服务地址/模型，并行处理
    if args.model_workers < 1 or args.model_workers > 8:
        parser.error("--model-workers must be between 1 and 8")
    if args.problem_timeout < 1:
        parser.error("--problem-timeout must be positive")
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if args.subshard_count < 1:
        parser.error("--subshard-count must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
