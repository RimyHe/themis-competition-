import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

from themis.core.validate.base import TestValidator, ValidationStats
from themis.utils.logger import logger
from themis.utils.pbar import get_progress_bar


class ValidationResult(NamedTuple):
    testcases: dict[str, list[str]]
    stats: ValidationStats | dict[str, Any]


def validate_single_problem(
    constraint_code: str,
    testcase_code: str,
    timeout: float | None = 10.0,
    output_dir: Path | None = None,
) -> ValidationResult:
    validator = TestValidator(constraint_code, testcase_code, timeout)
    try:
        filtered, stats = validator.validate_testcases()
        if output_dir:
            validator.save_results(filtered, output_dir)
        return ValidationResult(filtered, stats)
    except Exception as e:
        logger.error(f"Validate Single Problem error: {e}")
        return ValidationResult({}, {"error": str(e)})


def batch_validate_problems(
    problem_specs: dict[str, tuple[str, str]],
    max_workers: int = 96,
    timeout: float | None = 10.0,
) -> dict[str, ValidationResult]:
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                validate_single_problem,
                constraint_code=constraint,
                testcase_code=testgen,
                timeout=timeout,
            ): pid
            for pid, (constraint, testgen) in problem_specs.items()
        }

        with get_progress_bar() as pbar:
            main_task = pbar.add_task("[cyan]Validating...", total=len(problem_specs))
            for future in as_completed(future_map):
                pid = future_map[future]
                try:
                    result = future.result()
                    results[pid] = result
                except Exception as e:
                    logger.error(f"Task {pid} failed: {str(e)}")
                    results[pid] = ValidationResult({}, {"error": str(e)})
                finally:
                    pbar.update(main_task, advance=1)

    return results
