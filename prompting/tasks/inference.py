from typing import ClassVar
from prompting.rewards.reward import WeightedRewardModel, BaseRewardConfig

from prompting.tasks.base_task import BaseTextTask
from prompting.datasets.base import DatasetEntry
from prompting.llms.model_zoo import ModelConfig
import random
from prompting.llms.model_manager import model_manager
from vllm import SamplingParams
from abc import abstractmethod
from prompting.llms.model_zoo import ModelZoo


class InferenceRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = []


class BaseInferenceTask(BaseTextTask):
    query: str | None = None
    reference: str | None = None
    model: ModelConfig
    seed: int = random.randint(0, 1_000_000)

    @abstractmethod
    def make_query(self) -> str:
        raise NotImplementedError("Method make_query must be implemented")

    def make_reference(self) -> str:
        if self.model in model_manager.active_models.keys():
            model_manager.active_models[self.model].generate(self.query, SamplingParams(seed=self.seed))


class OrganicInferenceData(BaseInferenceTask):
    seed: int = random.randint(0, 1_000_000)

    def make_query(self) -> str:
        assert self.query is not None, "Organic Inference Tasks must be spawned with query"
        return self.query


class SyntheticInferenceTask(BaseInferenceTask):
    model: ModelConfig = ModelZoo.get_random()

    def make_query(self, dataset_entry: DatasetEntry) -> str:
        self.query = "What is the capital of France?"
        return self.query
