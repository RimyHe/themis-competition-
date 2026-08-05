"""Generate auditable small-input oracle candidates; execution happens elsewhere."""
# 提供oracle的通用提示词， 生成Oracle的封装，静态安全检查
import ast

from themis.core.completion.generator import LLMGenerator
from themis.core.completion.utils import GenerationConfig
from themis.utils.sanitize import extract_code_block

from .oracle_consensus import OracleCandidate


ORACLE_PROMPT = """Write a self-contained Python ACM stdin/stdout oracle for this programming problem.
It is only used for small inputs, so prefer a direct, obviously correct brute-force or simulation algorithm.
Do not read files, use the network, start processes, or add explanations. Return exactly one fenced Python block.

Problem:
{question}
"""


def static_check_oracle(code: str) -> None:
    tree = ast.parse(code)
    forbidden = {"os", "random", "subprocess", "socket", "requests", "urllib", "pathlib"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {alias.name.split(".")[0] for alias in node.names}
            if names & forbidden:
                raise ValueError(f"oracle imports forbidden modules: {sorted(names & forbidden)}")


class OracleGenerator:
    def __init__(self, config: GenerationConfig):
        self._llm = LLMGenerator(config)

    def generate_one(self, question: str, algorithm_label: str) -> OracleCandidate:
        prompt = ORACLE_PROMPT.replace("{question}", question)
        raw_response = self._llm.fast_completion(prompt)
        code = extract_code_block(raw_response)
        static_check_oracle(code)
        return OracleCandidate(code=code, raw_response=raw_response, algorithm_label=algorithm_label)
