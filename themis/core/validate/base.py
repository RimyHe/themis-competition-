import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import orjson

from themis.utils.logger import logger

ENABLE_VALIDATION = True
MAX_TESTCASES = 32


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


@dataclass
class ValidationStats:
    generated: dict[str, int] = field(default_factory=dict)
    filtered: dict[str, int] = field(default_factory=dict)


class TestValidator:
    def __init__(self, constraint_code: str, testgen_code: str, timeout: float = 5.0):
        self.constraint = self._load_component(constraint_code, "ProblemConstraint")
        self.testgen = self._load_component(testgen_code, "TestGenerator")
        self.timeout = timeout

    def _load_component(self, source: str, component_name: str) -> Any:
        module = ModuleType(f"dynamic_{hash(source)}")
        sys.modules[module.__name__] = module
        try:
            exec(compile(source, "<string>", "exec"), module.__dict__)
            return getattr(module, component_name)()
        except AttributeError:
            logger.error(f"Missing {component_name} in source")
            return None
        except Exception as e:
            logger.error(f"Error loading {component_name}: {e}")
            return None
        finally:
            sys.modules.pop(module.__name__, None)

    def call_with_timeout(self, func: Callable) -> list[str]:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.timeout * 2))
        try:
            return func()
        except TimeoutError:
            logger.warning(f"Generation timeout for {func.__name__}")
            return []
        except Exception:
            pass
            return []
        finally:
            signal.alarm(0)

    def _batch_validate(self, inputs: list[str]) -> list[str]:
        if not ENABLE_VALIDATION:
            return inputs
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.timeout * 2))
        valid_inputs = []
        for input in inputs:
            try:
                valid = self.constraint.validate(input)
                if valid:
                    valid_inputs.append(input)
            except TimeoutError:
                logger.warning("Validation timeout")
                break
            except Exception:
                pass
        return valid_inputs

    def validate_testcases(self) -> tuple[dict, ValidationStats]:
        generated = {
            "base": self.call_with_timeout(self.testgen.generate_base_case)[:MAX_TESTCASES],
            "corner": self.call_with_timeout(self.testgen.generate_corner_case)[:MAX_TESTCASES],
            "complex": self.call_with_timeout(self.testgen.generate_complex_case)[:MAX_TESTCASES],
        }

        stats = ValidationStats(generated={k: len(v) for k, v in generated.items()})
        filtered = {k: self._batch_validate(v) for k, v in generated.items()}
        stats.filtered = {k: len(v) for k, v in filtered.items()}

        return filtered, stats

    def save_results(self, testcases: dict, output_path: Path) -> Path:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        output_path.write_bytes(orjson.dumps(testcases, option=orjson.OPT_INDENT_2))
        return output_path
