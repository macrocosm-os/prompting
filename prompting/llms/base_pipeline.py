from abc import ABC
from typing import Any


class BasePipeline(ABC):
    def __call__(self, system_prompt:str, prompt:str, **kwargs: dict) -> Any:
        ...
        

