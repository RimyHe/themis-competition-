import asyncio
from datetime import UTC, datetime

from themis.core.completion.utils import GenerationConfig, OpenAICompletion
from themis.utils.logger import logger
from themis.utils.pbar import get_progress_bar


class LLMGenerator:
    r"""Class for generating responses via OpenAI Compatible APIs."""

    def __init__(self, config: GenerationConfig, system_prompt: str | None = None) -> None:
        self.config = config
        self.client = OpenAICompletion(config.base_url, config.api_key)
        self.system_prompt = system_prompt

    async def generate_sample(
        self, prompt: str, task_id: str | None = None, sample_id: int | None = None
    ) -> tuple[str, int, str | None]:
        r"""generate a single sample asynchronously."""
        messages = (
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            if self.system_prompt
            else [{"role": "user", "content": prompt}]
        )

        try:
            response = await self.client.completion(
                is_chat=self.config.is_chat,
                model=self.config.model,
                prompt=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                stop=self.config.stop,
                timeout=self.config.timeout,
            )
            return (task_id, sample_id, response)
        except Exception as e:
            logger.error(f"Error processing task {task_id} sample {sample_id}: {str(e)}")
            return (task_id, sample_id, None)

    async def generate_sample_trace(
        self, prompt: str, task_id: str | None = None, sample_id: int | None = None
    ) -> dict:
        r"""Generate a sample and return prompt, raw response, and request metadata."""
        messages = (
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            if self.system_prompt
            else [{"role": "user", "content": prompt}]
        )
        started_at = datetime.now(UTC).isoformat()
        details = await self.client.completion_details(
            is_chat=self.config.is_chat,
            model=self.config.model,
            prompt=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            stop=self.config.stop,
            timeout=self.config.timeout,
        )
        ended_at = datetime.now(UTC).isoformat()
        return {
            "task_id": task_id,
            "sample_id": sample_id,
            "prompt": messages,
            "request_started_at": started_at,
            "request_ended_at": ended_at,
            "model": self.config.model,
            "sampling": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "stop": self.config.stop,
                "timeout": self.config.timeout,
            },
            "response_text": details.content,
            "finish_reason": details.finish_reason,
            "raw_response": details.raw_response,
            "error": details.error,
        }

    def fast_completion(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.generate_sample(prompt))[2]

    async def async_batch_completion(
        self, prompts: list[str], task_ids: list[str] | None = None, max_concurrent: int = 16
    ) -> dict[str, str]:
        r"""Generate completions for multiple prompts in batch with concurrency control."""
        if not task_ids:
            task_ids = [str(i) for i in range(len(prompts))]

        if len(prompts) != len(task_ids):
            raise ValueError("Number of prompts must match number of task_ids")

        semaphore = asyncio.Semaphore(max_concurrent)

        with get_progress_bar() as pbar:
            main_task = pbar.add_task("[cyan]Generating completions", total=len(prompts))

            async def bounded_generate(prompt: str, task_id: str) -> tuple[str, str]:
                async with semaphore:
                    _, _, result = await self.generate_sample(prompt, task_id)
                    pbar.update(main_task, advance=1)
                    return task_id, result or ""

            tasks = [bounded_generate(prompt, task_id) for prompt, task_id in zip(prompts, task_ids, strict=False)]
            results = await asyncio.gather(*tasks)
        return dict(results)

    def batch_completion(
        self, prompts: list[str], task_ids: list[str] | None = None, max_workers: int = 16
    ) -> dict[str, str]:
        r"""Synchronous wrapper for batch completion."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_batch_completion(prompts, task_ids, max_workers))

    async def async_batch_completion_trace(
        self, prompts: list[str], task_ids: list[str] | None = None, max_concurrent: int = 16
    ) -> dict[str, dict]:
        r"""Generate completions while preserving prompt and response metadata."""
        if not task_ids:
            task_ids = [str(i) for i in range(len(prompts))]

        if len(prompts) != len(task_ids):
            raise ValueError("Number of prompts must match number of task_ids")

        semaphore = asyncio.Semaphore(max_concurrent)

        with get_progress_bar() as pbar:
            main_task = pbar.add_task("[cyan]Generating completions", total=len(prompts))

            async def bounded_generate(prompt: str, task_id: str) -> tuple[str, dict]:
                async with semaphore:
                    trace = await self.generate_sample_trace(prompt, task_id)
                    pbar.update(main_task, advance=1)
                    return task_id, trace

            tasks = [bounded_generate(prompt, task_id) for prompt, task_id in zip(prompts, task_ids, strict=False)]
            results = await asyncio.gather(*tasks)
        return dict(results)

    def batch_completion_trace(
        self, prompts: list[str], task_ids: list[str] | None = None, max_workers: int = 16
    ) -> dict[str, dict]:
        r"""Synchronous wrapper for traced batch completion."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_batch_completion_trace(prompts, task_ids, max_workers))
