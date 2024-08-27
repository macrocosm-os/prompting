from abc import ABC, abstractmethod
from prompting.utils.cleaners import CleanerPipeline
from typing import Any
from loguru import logger
from pydantic import BaseModel
from vllm import LLM


class BasePipeline(ABC, BaseModel):
    @abstractmethod
    def __call__(self, composed_prompt: str, **kwargs: dict) -> Any: ...


class BaseLLM(ABC):
    llm: LLM
    model_kwargs: dict
    system_prompt: str | None = None
    messages: list[dict] = []
    times: list[int] = []
    tokenizer: Any = None

    def __init__(
        self,
        llm_pipeline: BasePipeline,
        system_prompt: str,
        model_kwargs: dict,
    ):
        self.llm = llm_pipeline
        self.system_prompt = system_prompt
        self.model_kwargs = model_kwargs
        self.messages = []
        self.times = []
        self.tokenizer = None

    @abstractmethod
    def query(
        self,
        message: str,
        role: str = "user",
        cleaner: CleanerPipeline = None,
    ) -> str: ...

    def forward(self, messages: list[dict[str, str]]) -> str:
        return self._forward(messages)

    @abstractmethod
    def _forward(self, messages: list[dict[str, str]]) -> str: ...

    def clean_response(self, cleaner: CleanerPipeline, response: str) -> str:
        clean_response = response
        if cleaner is not None:
            clean_response = cleaner.apply(generation=response)
            if clean_response != response:
                logger.debug(f"Response cleaned, chars removed: {len(response) - len(clean_response)}...")

        return clean_response
