from typing import ClassVar

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict

from prompting.settings import settings


class ModelConfig(BaseModel):
    llm_model_id: str
    reward: float
    min_ram: float
    model_config = ConfigDict(frozen=True)

    def __hash__(self):
        return hash((self.llm_model_id, self.reward, self.min_ram))


class ModelZoo:
    # Currently, we are only using one single model - the one the validator is running
    models_configs: ClassVar[list[ModelConfig]] = [
        ModelConfig(
            llm_model_id=settings.LLM_MODEL,
            reward=1,
            min_ram=settings.MAX_ALLOWED_VRAM_GB,
        ),
    ]

    # Code below can be uncommended for testing purposes and demonstrates how we rotate multiple LLMs in the future
    # models_configs: ClassVar[list[ModelConfig]] = [
    # ModelConfig(model_id="casperhansen/mistral-nemo-instruct-2407-awq", reward=0.1, min_ram=24),
    # ModelConfig(model_id="casperhansen/qwen2-0.5b-instruct-awq", reward=0.1, min_ram=10),
    # ]

    @classmethod
    def get_all_models(cls) -> list[str]:
        return [model.llm_model_id for model in cls.models_configs]

    @classmethod
    def get_random(cls, max_ram: float = np.inf) -> ModelConfig:
        models = [model for model in cls.models_configs if model.min_ram <= max_ram]
        if len(models) == 0:
            raise Exception(f"No model with < {max_ram}GB memory requirements found")
        return np.random.choice(models)

    @classmethod
    def get_model_by_id(cls, model_id: str) -> ModelConfig:
        try:
            return [model for model in cls.models_configs if model.llm_model_id == model_id][0]
        except Exception as ex:
            logger.error(f"Model {model_id} not found in ModelZoo: {ex}")
