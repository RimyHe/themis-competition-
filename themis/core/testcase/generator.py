import json
from pathlib import Path

from themis.core.completion.generator import LLMGenerator
from themis.core.completion.utils import GenerationConfig
from themis.utils.sanitize import extract_code_block


def _write_once(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path
    attempt = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.retry{attempt}{path.suffix}")
        attempt += 1
    candidate.write_text(content, encoding="utf-8")
    return candidate


def _validate_raw_response(problem_id: str, trace: dict) -> str:
    if trace.get("error"):
        raise RuntimeError(f"{problem_id}: model request failed: {trace['error']}")
    if trace.get("finish_reason") == "length":
        raise RuntimeError(f"{problem_id}: model response stopped because finish_reason=length")
    raw_response = trace.get("response_text") or ""
    if not raw_response.strip():
        raise RuntimeError(f"{problem_id}: model response is empty")
    if "```" not in raw_response:
        raise RuntimeError(f"{problem_id}: model response does not contain a fenced code block")
    code = extract_code_block(raw_response)
    if not code.strip():
        raise RuntimeError(f"{problem_id}: extracted code is empty")
    return code


def _save_trace_artifacts(stage: str, problem_id: str, trace: dict, code: str | None, output_dir: str) -> None:
    base = Path(output_dir)
    _write_once(base / "prompts" / stage / f"{problem_id}.json", json.dumps(trace.get("prompt"), indent=2))
    _write_once(base / "raw_responses" / stage / f"{problem_id}.txt", trace.get("response_text") or "")
    metadata = {k: v for k, v in trace.items() if k not in {"prompt", "response_text"}}
    _write_once(base / "generation_metadata" / stage / f"{problem_id}.json", json.dumps(metadata, indent=2))
    if code is not None:
        _write_once(base / "generated_code" / stage / f"{problem_id}.py", code)


class TestCaseGenerator:
    r"""Utility class for generating and managing test cases."""

    def __init__(self, config: GenerationConfig):
        self.llm = LLMGenerator(config)
        self.template = (Path(__file__).parent / "template.txt").read_text()

    def generate_testcase_code(self, problem_statement: str, solution: str) -> str:
        prompt = self.template.replace("{problem_statement}", problem_statement)
        prompt = prompt.replace("{solution}", solution)
        raw_response = self.llm.fast_completion(prompt)
        code = extract_code_block(raw_response)
        return code

    def batch_generate_testcase_code(
        self,
        problem_statements: dict[str, str],
        solutions: dict[str, str],
        max_workers: int = 16,
    ) -> dict[str, str]:
        problem_ids = list(problem_statements.keys())
        prompts = []

        for problem_id in problem_ids:
            prompt = self.template.replace("{problem_statement}", problem_statements[problem_id])
            prompt = prompt.replace("{solution}", solutions[problem_id])
            prompts.append(prompt)

        raw_responses = self.llm.batch_completion(prompts=prompts, task_ids=problem_ids, max_workers=max_workers)

        results = {}
        for problem_id, raw_response in raw_responses.items():
            code = extract_code_block(raw_response)
            results[problem_id] = code

        return results

    def batch_generate_testcase_code_traced(
        self,
        problem_statements: dict[str, str],
        solutions: dict[str, str],
        max_workers: int = 16,
        output_dir: str = "outputs",
    ) -> dict[str, str]:
        problem_ids = list(problem_statements.keys())
        prompts = []

        for problem_id in problem_ids:
            prompt = self.template.replace("{problem_statement}", problem_statements[problem_id])
            prompt = prompt.replace("{solution}", solutions[problem_id])
            prompts.append(prompt)

        traces = self.llm.batch_completion_trace(prompts=prompts, task_ids=problem_ids, max_workers=max_workers)
        results = {}
        errors = []
        for problem_id, trace in traces.items():
            code = None
            try:
                code = _validate_raw_response(problem_id, trace)
                results[problem_id] = code
            except RuntimeError as exc:
                errors.append(str(exc))
            finally:
                _save_trace_artifacts("stage2", problem_id, trace, code, output_dir)

        if errors:
            raise RuntimeError("; ".join(errors))

        return results

    def save_testcase_code(self, code: str, output_path: str) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(code)
        return str(output_path)

    def batch_save_testcase_code(
        self,
        codes: dict[str, str],
        output_dir: str,
        filename_template: str = "{problem_id}_testcase.py",
    ) -> dict[str, str]:
        r"""Save test case code for multiple problems."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        for problem_id, code in codes.items():
            filename = filename_template.format(problem_id=problem_id)
            output_path = output_dir / filename

            with open(output_path, "w") as f:
                f.write(code)

            results[problem_id] = str(output_path)

        return results
