from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class ValidationError(Exception):
    """Exception raised for validation errors"""

    pass


class ConstraintBase(ABC):
    def __init__(self) -> None:
        self.max_value: float = float("inf")
        self.min_value: float = -float("inf")
        self.max_length: int = 10**6
        self.max_lines: int = 10**5
        self.allowed_chars: set[str] | None = None

    @abstractmethod
    def validate_structure(self, raw_input: str) -> bool:
        pass

    @abstractmethod
    def validate_semantic(self, parsed_input: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def parse_input(self, raw_input: str) -> dict[str, Any] | None:
        pass

    def check_numeric(
        self,
        value: int | float,
        min_val: float | None = None,
        max_val: float | None = None,
        inclusive: bool = True,
    ) -> bool:
        min_val = self.min_value if min_val is None else min_val
        max_val = self.max_value if max_val is None else max_val

        if inclusive:
            return min_val <= value <= max_val
        return min_val < value < max_val

    def check_length(self, container: Iterable[Any], max_len: int | None = None, min_len: int = 0) -> bool:
        max_len = self.max_length if max_len is None else max_len
        return min_len <= len(container) <= max_len

    def check_char_set(self, text: str, allowed_chars: set[str] | None = None) -> bool:
        if allowed_chars is None:
            allowed_chars = self.allowed_chars

        if allowed_chars is None:
            return True

        return all(char in allowed_chars for char in text)
