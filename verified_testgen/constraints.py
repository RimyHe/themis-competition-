"""Themis Stage 1--3 adapter for constraint-aware competition inputs."""

import ast
import multiprocessing
import re
import resource
import signal
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .input_generation import CandidateInput, filter_candidates, interleave_categories

if TYPE_CHECKING:
    from themis.core.completion.generator import LLMGenerator
    from themis.core.completion.utils import GenerationConfig


@dataclass(frozen=True)
class ConstraintArtifact:
    code: str


class ConstraintService:
    def __init__(self, config: "GenerationConfig"):
        from themis.core.constraint.generator import ConstraintGenerator

        self._generator = ConstraintGenerator(config)

    def generate(self, question: str, oracle_hint: str = "") -> ConstraintArtifact:
        return ConstraintArtifact(code=self._generator.generate_constraint_code(question, oracle_hint))


def filter_valid_inputs(inputs: list[str], validator: Callable[[str], bool], max_input_bytes: int) -> list[str]:
    """Filter using an externally supplied, isolated validator execution path."""
    accepted = []
    for test_input in inputs:
        if len(test_input.encode("utf-8")) <= max_input_bytes and validator(test_input):
            accepted.append(test_input)
    return accepted


@dataclass(frozen=True)
class ThemisGenerationArtifacts:
    """Auditable source and trace data emitted by Themis stages 1--3."""

    constraint_code: str | None
    testgen_code: str | None
    constraint_trace: dict[str, Any]
    testgen_trace: dict[str, Any]
    validation_stats: dict[str, Any]
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _template(filename: str) -> str:
    return (Path(__file__).resolve().parents[1] / "themis" / "core" / filename).read_text(encoding="utf-8")


CONSTRAINT_TEMPLATE = _template("constraint/template.txt")
TESTGEN_TEMPLATE = _template("testcase/template.txt")
NO_REFERENCE_SOLUTION = (
    "No trusted reference solution is available. Derive the stdin schema and legal input domain from the statement only."
)


def _themis_prompt(template: str, question: str) -> str:
    """Use Themis's templates without treating an untrusted candidate as ground truth."""
    return template.replace("{problem_statement}", question).replace("{solution}", NO_REFERENCE_SOLUTION)


def _extract_component(trace: dict[str, Any], component_name: str) -> str:
    if trace.get("error"):
        raise ValueError(f"model request failed: {trace['error']}")
    if trace.get("finish_reason") == "length":
        raise ValueError("model response stopped because finish_reason=length")
    response = trace.get("response_text") or ""
    if not response.strip():
        raise ValueError("model response is empty")
    code = _extract_code_block(response)
    tree = ast.parse(code)
    component = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == component_name), None)
    if component is None:
        raise ValueError(f"generated code does not define {component_name}")
    required_methods = {
        "ProblemConstraint": {"parse_input", "validate"},
        "TestGenerator": {"generate_base_case", "generate_corner_case", "generate_complex_case"},
    }[component_name]
    defined_methods = {node.name for node in component.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if missing := required_methods - defined_methods:
        raise ValueError(f"{component_name} is missing methods: {sorted(missing)}")
    return code


class ThemisInputPipeline:
    """Generate and validate inputs through Themis before oracle labelling.

    The dynamically generated components are executed by Themis's existing Stage 3
    validator.  Production runs must therefore use the dedicated, credential-free
    execution environment documented for this competition.
    """

    def __init__(self, generator: "LLMGenerator", validation_timeout: float = 5.0, max_input_bytes: int = 65536):
        self._generator = generator
        self._validation_timeout = validation_timeout
        self._max_input_bytes = max_input_bytes

    async def generate_candidates(
        self, question: str
    ) -> tuple[list[CandidateInput], ThemisGenerationArtifacts]:
        constraint_trace = await self._generator.generate_sample_trace(
            _themis_prompt(CONSTRAINT_TEMPLATE, question), task_id="themis-constraint"
        )
        testgen_trace = await self._generator.generate_sample_trace(
            _themis_prompt(TESTGEN_TEMPLATE, question), task_id="themis-testgen"
        )
        try:
            constraint_code = _extract_component(constraint_trace, "ProblemConstraint")
            testgen_code = _extract_component(testgen_trace, "TestGenerator")
            try:
                result = _validate_single_problem_isolated(
                    constraint_code, testgen_code, self._validation_timeout
                )
            finally:
                # Themis's _batch_validate leaves its process-wide SIGALRM armed.
                signal.alarm(0)
            if not isinstance(result.stats, dict) and not is_dataclass(result.stats):
                raise ValueError("Themis validation returned malformed statistics")
            stats = asdict(result.stats) if is_dataclass(result.stats) else result.stats
            if isinstance(stats, dict) and stats.get("error"):
                raise ValueError(f"Themis validation failed: {stats['error']}")
            if not isinstance(result.testcases, dict):
                raise ValueError("Themis validation returned malformed test cases")
            generated = [
                CandidateInput(value, category, f"themis-{category}")
                for category in ("base", "corner", "complex")
                for value in result.testcases.get(category, [])
                if isinstance(value, str)
            ]
            candidates = filter_candidates(generated, lambda _: True, self._max_input_bytes)
            return candidates, ThemisGenerationArtifacts(
                constraint_code, testgen_code, constraint_trace, testgen_trace, stats
            )
        except (SyntaxError, ValueError, TypeError) as exc:
            return [], ThemisGenerationArtifacts(
                None, None, constraint_trace, testgen_trace, {}, str(exc)
            )


def balanced_themis_candidates(candidates: list[CandidateInput], limit: int) -> list[CandidateInput]:
    """Bound Stage 3 output while retaining all Themis generation categories."""
    return interleave_categories(candidates, limit)


def _validate_single_problem(constraint_code: str, testgen_code: str, timeout: float) -> Any:
    """Delay Themis imports so local adapter tests need no model client installation."""
    from themis.core.validate import validate_single_problem

    return validate_single_problem(constraint_code, testgen_code, timeout=timeout)


_VALIDATION_MEMORY_LIMIT_BYTES = 2 * 1024**3
_VALIDATION_WALL_TIMEOUT_SECONDS = 15


def _validation_worker(
    connection: Any, constraint_code: str, testgen_code: str, timeout: float
) -> None:
    """Run untrusted generated code outside the long-lived shard worker."""
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (_VALIDATION_MEMORY_LIMIT_BYTES, _VALIDATION_MEMORY_LIMIT_BYTES),
        )
        result = _validate_single_problem(constraint_code, testgen_code, timeout)
        connection.send((result.testcases, result.stats))
    except BaseException as exc:
        connection.send(({}, {"error": f"isolated validation failed: {exc}"}))
    finally:
        connection.close()


def _validate_single_problem_isolated(constraint_code: str, testgen_code: str, timeout: float) -> Any:
    """Bound generated test-case execution so one pathological problem cannot OOM the shard."""
    from themis.core.validate import ValidationResult

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_validation_worker,
        args=(sender, constraint_code, testgen_code, timeout),
    )
    process.start()
    sender.close()
    process.join(_VALIDATION_WALL_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join()
        receiver.close()
        return ValidationResult({}, {"error": "isolated validation timed out"})
    try:
        if receiver.poll():
            testcases, stats = receiver.recv()
            return ValidationResult(testcases, stats)
        return ValidationResult({}, {"error": "isolated validation exited without a result"})
    finally:
        receiver.close()


def _extract_code_block(response: str) -> str:
    """Match Themis's fenced-code extraction without importing the model client stack."""
    response = response.strip().split("</think>")[-1].strip()
    if "```" not in response:
        return response
    matches = re.findall(r"```(.*?)\n([\s\S]*?)\n```", response)
    return matches[0][1] if matches else response
