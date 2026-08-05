"""Merge completed full-run shards into the competition submission order."""

import argparse
import json
from pathlib import Path

from .io import load_problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge non-overlapping competition run shards.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    missing = []
    for index, problem in enumerate(load_problems(args.input)):
        filename = f"test_cases_{problem.problem_id}.jsonl"
        # Run directories are supplied in shard-index order.  This deliberately
        # ignores pre-sharding artifacts from another shard's directory.
        expected = args.run_dir[index % len(args.run_dir)] / problem.problem_id / "output" / filename
        if not expected.is_file():
            missing.append({"problem_id": problem.problem_id, "expected": str(expected)})
            continue
        records.append(json.loads(expected.read_text(encoding="utf-8")))

    if missing:
        raise SystemExit(f"cannot merge: {len(missing)} missing or duplicated problem outputs: {missing[:3]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"merged {len(records)} records into {args.output}")


if __name__ == "__main__":
    main()
