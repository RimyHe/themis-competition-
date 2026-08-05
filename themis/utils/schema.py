import orjson
from pydantic import BaseModel


class Problem(BaseModel):
    id: str
    description: str | None = None
    constraint_code: str | None = None
    testgen_code: str | None = None
    original_testcases: list[dict] | None = None
    generated_testcases: dict[str, list[str]] | list[dict] | None = None
    solution: str | None = None

    def orjson_dumps(self, **kwargs):
        return orjson.dumps(self.model_dump(), **kwargs)


def save_problems(problems: dict[str, Problem], file_path: str):
    data = {k: v.model_dump() for k, v in problems.items()}
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def load_problems(file_path: str) -> dict[str, Problem]:
    with open(file_path, "rb") as f:
        data = orjson.loads(f.read())
    return {k: Problem.model_validate(v) for k, v in data.items()}
