"""Continuously collect batch warnings and per-problem failures into JSONL."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path


MARKERS = (" WARNING ", " ERROR ", "Traceback", "Max tokens exceeded")


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_record(destination: Path, record: dict) -> None:
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def collect_log(log_path: Path, destination: Path, offset: int) -> int:
    if not log_path.exists():
        return offset
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        for line in handle:
            if any(marker in line for marker in MARKERS):
                write_record(destination, {"at": now(), "kind": "log", "message": line.rstrip()[:4000]})
        return handle.tell()


def collect_summaries(run_dir: Path, destination: Path, seen: set[str]) -> None:
    for summary_path in run_dir.glob("*/summary.json"):
        problem_id = summary_path.parent.name
        if problem_id in seen:
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            write_record(destination, {"at": now(), "kind": "summary_parse_error", "problem_id": problem_id, "error": str(exc)})
            continue
        seen.add(problem_id)
        if summary.get("error"):
            write_record(destination, {"at": now(), "kind": "problem_error", "problem_id": problem_id, "error": summary["error"]})
        elif summary.get("selected_cases", 0) == 0:
            write_record(
                destination,
                {
                    "at": now(),
                    "kind": "empty_submission",
                    "problem_id": problem_id,
                    "oracle_candidates": summary.get("oracle_candidates"),
                    "candidate_inputs": summary.get("candidate_inputs"),
                    "consensus_cases": summary.get("consensus_cases"),
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect diagnostics for a running competition batch.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    destination = args.run_dir / "diagnostics.jsonl"
    offset = 0
    seen: set[str] = set()
    while Path(f"/proc/{args.pid}").exists():
        offset = collect_log(args.run_dir / "run.log", destination, offset)
        collect_summaries(args.run_dir, destination, seen)
        (args.run_dir / "diagnostics_status.json").write_text(
            json.dumps({"updated_at": now(), "completed_summaries": len(seen)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        time.sleep(args.interval)
    offset = collect_log(args.run_dir / "run.log", destination, offset)
    collect_summaries(args.run_dir, destination, seen)


if __name__ == "__main__":
    main()
