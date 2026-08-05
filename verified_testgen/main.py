"""CLI entry point for selecting prepared, consensus-labelled competition cases."""

import argparse
from pathlib import Path

from .io import load_prepared_cases, load_problems, write_submission
from .selection import select_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Write validated <=5-case competition submissions.")
    parser.add_argument("--input", type=Path, required=True, help="Competition JSON array")
    parser.add_argument("--prepared-cases", type=Path, required=True, help="Consensus-labelled candidate case JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N problems")
    args = parser.parse_args()

    problems = load_problems(args.input)
    if args.limit is not None:
        problems = problems[: args.limit]
    prepared = load_prepared_cases(args.prepared_cases)
    for problem in problems:
        write_submission(args.output_dir, problem, select_cases(prepared.get(problem.problem_id, [])))


if __name__ == "__main__":
    main()
