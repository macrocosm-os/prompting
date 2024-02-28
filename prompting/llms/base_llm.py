from abc import ABC, abstractmethod
from prompting.cleaners.cleaner import CleanerPipeline
from typing import Any


class BasePipeline(ABC):
    @abstractmethod
    def __call__(self, system_prompt: str, prompt: str, **kwargs: dict) -> Any:
        ...


class BaseLLM(ABC):
    def __init__(
        self,
        llm_pipeline: BasePipeline,
        system_prompt: str,
        max_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ):
        self.llm_pipeline = llm_pipeline
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.messages = []

    def query(
        self,
        message: str,
        role: str = "user",
        disregard_system_prompt: bool = False,
        cleaner: CleanerPipeline = None,
    ) -> str:
        ...
