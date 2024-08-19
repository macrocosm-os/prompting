from typing import ClassVar
import numpy as np
from pydantic import BaseModel, ConfigDict
from loguru import logger


class ModelConfig(BaseModel):
    model_id: str
    reward: float
    min_ram: float
    model_config = ConfigDict(frozen=True)

    def __hash__(self):
        return hash((self.model_id, self.reward, self.min_ram))


class ModelZoo:
    models_configs: ClassVar[list[ModelConfig]] = [
        # ModelConfig(model_id="casperhansen/mistral-nemo-instruct-2407-awq", reward=0.1, min_ram=24),
        ModelConfig(model_id="casperhansen/llama-3-8b-instruct-awq", reward=0.1, min_ram=24),
        # ModelConfig(model_id="casperhansen/llama-3-70b-instruct-awq", reward=0.8, min_ram=70),
    ]

    @classmethod
    def get_all_models(cls) -> list[str]:
        return [model.model_id for model in cls.models_configs]

    @classmethod
    def get_random(cls, max_ram: float = np.inf) -> ModelConfig:
        models = [model for model in cls.models_configs if model.min_ram <= max_ram]
        return np.random.choice(models)

    @classmethod
    def get_model_by_id(cls, model_id: str) -> ModelConfig:
        try:
            return [model for model in cls.models_configs if model.model_id == model_id][0]
        except Exception as ex:
            logger.error(f"Model {model_id} not found in ModelZoo: {ex}")
