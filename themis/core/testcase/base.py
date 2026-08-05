import random
import string
from abc import ABC, abstractmethod


class TestGeneratorBase(ABC):
    def __init__(
        self,
        seed: int = 42,
        num_base_cases: int = 30,
        num_corner_cases: int = 10,
        num_complex_cases: int = 60,
    ):
        self.seed = seed
        random.seed(self.seed)
        self.num_base_cases = num_base_cases
        self.num_corner_cases = num_corner_cases
        self.num_complex_cases = num_complex_cases

    @abstractmethod
    def generate_base_case(self) -> list[str]:
        pass

    @abstractmethod
    def generate_corner_case(self) -> list[str]:
        pass

    @abstractmethod
    def generate_complex_case(self) -> list[str]:
        pass

    def generate_all_cases(self) -> dict[str, list[str]]:
        return {
            "base": self.generate_base_case(),
            "corner": self.generate_corner_case(),
            "complex": self.generate_complex_case(),
        }

    def gen_int(self, min_val: int = -(10**9), max_val: int = 10**9) -> int:
        return random.randint(int(min_val), int(max_val))

    def gen_float(self, min_val: float = -(10**9), max_val: float = 10**9, precision: int = 6) -> float:
        val = random.uniform(min_val, max_val)
        return round(val, precision)

    def gen_string(self, length: int, charset: str = string.ascii_lowercase) -> str:
        return "".join(random.choices(charset, k=length))
