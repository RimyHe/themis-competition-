# Verified Test Generation

Verified Test Generation is a research prototype for producing high-value test cases for programming problems. It turns a natural-language problem statement into a small, auditable set of input/output cases that can expose incorrect candidate solutions.

## Background

Hand-written tests rarely cover the full constraint space of a programming problem. Large language models can propose useful edge cases, but generated inputs and expected outputs must be checked before they can be trusted. This project treats test generation as a verification pipeline rather than a one-shot prompting task.

## Approach

For each problem, the pipeline:

1. extracts an executable input constraint and generates base, corner, and complex inputs;
2. validates generated inputs in an isolated process;
3. asks for several deliberately independent small-input reference implementations;
4. accepts expected outputs only when reference implementations reach consensus;
5. executes accepted cases against candidate solutions and selects at most five cases with broad error-detection coverage.

The result includes intermediate artifacts such as generated constraints, reference implementations, execution results, and the final selected cases, so each decision can be inspected after a run.

## Requirements

- Python 3.10 or later
- An OpenAI-compatible text-generation endpoint
- A `mini-judge`-compatible execution service for running candidate and reference programs

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running

The input is a JSON array of problems. Each record must provide an `id`, a `question`, and a `solutions` array containing candidate programs.

```bash
python -m verified_testgen.pilot \
  --input problems.json \
  --output-dir outputs \
  --model-url http://127.0.0.1:30000/v1 \
  --judge-url http://127.0.0.1:8000/api/v1/judge \
  --model-server-gpu 0
```

For a resumable sharded run over a larger dataset:

```bash
python -m verified_testgen.full_run \
  --input problems.json \
  --output-dir outputs \
  --model-server-gpu 0 \
  --shard-index 0 \
  --shard-count 1
```

## Tests

```bash
pytest
```

## Safety

Generated constraint and test-generator code is executed in a separate process with time and memory limits. Reference implementations and candidate solutions are sent to the configured judge service; run that service in an isolated environment appropriate for untrusted code.

## License

MIT. See [LICENSE](LICENSE).
