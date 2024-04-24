import bittensor as bt
from abc import ABC, abstractmethod
from prompting.cleaners.cleaner import CleanerPipeline
from typing import Any, Dict, List


class BasePipeline(ABC):
    @abstractmethod
    def __call__(self, composed_prompt: str, **kwargs: dict) -> Any: ...


class BaseLLM(ABC):
    def __init__(
        self,
        llm_pipeline: BasePipeline,
        system_prompt: str,
        model_kwargs: dict,
    ):
        self.llm_pipeline = llm_pipeline
        self.system_prompt = system_prompt
        self.model_kwargs = model_kwargs
        self.messages = []
        self.times = []

    def query(
        self,
        message: str,
        role: str = "user",
        disregard_system_prompt: bool = False,
        cleaner: CleanerPipeline = None,
    ) -> str: ...

    def forward(self, messages: List[Dict[str, str]]): ...

    def clean_response(self, cleaner: CleanerPipeline, response: str) -> str:
        if cleaner is not None:
            clean_response = cleaner.apply(generation=response)
            if clean_response != response:
                bt.logging.debug(
                    f"Response cleaned, chars removed: {len(response) - len(clean_response)}..."
                )

            return clean_response
        return response
