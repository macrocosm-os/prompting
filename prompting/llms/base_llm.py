import bittensor as bt
from abc import ABC, abstractmethod
from prompting.cleaners.cleaner import CleanerPipeline
from typing import Any, Dict, List, Optional, Union


class BasePipeline(ABC):
    @abstractmethod
    def __call__(self, composed_prompt: str, **kwargs: dict) -> Any:
        ...


class BaseLLM(ABC):
    def __init__(
        self,
        llm_pipeline: BasePipeline,
        system_prompt: str,
        model_kwargs: dict[str, Union[int, float]],
    ):
        self.llm_pipeline: BasePipeline = llm_pipeline
        self.system_prompt: str = system_prompt
        self.model_kwargs: dict[str, Union[int, float]] = model_kwargs
        self.messages: list[str] = []
        self.times: list[float] = []
        self.tokenizer = None

    def query(
        self,
        message: str,
        role: str = "user",
        disregard_system_prompt: Optional[bool] = False,
        cleaner: CleanerPipeline = None,
    ) -> str:
        ...

    def forward(self, messages: List[Dict[str, str]]):
        ...

    def clean_response(self, cleaner: CleanerPipeline, response: str) -> str:
        if cleaner is not None:
            clean_response = cleaner.apply(generation=response)
            if clean_response != response:
                bt.logging.debug(
                    f"Response cleaned, chars removed: {len(response) - len(clean_response)}..."
                )

            return clean_response
        return response
