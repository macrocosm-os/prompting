from typing import ClassVar
from prompting.rewards.reward import WeightedRewardModel, BaseRewardConfig
from prompting.rewards.exact_match import ExactMatchRewardModel

from prompting.tasks.base_task import BaseTextTask
from prompting.datasets.base import DatasetEntry
from prompting.llms.model_zoo import ModelConfig
import random
from prompting.llms.model_manager import model_manager
from vllm import SamplingParams
from abc import abstractmethod
from prompting.llms.model_zoo import ModelZoo
from prompting.datasets.mmlu import MMLUEntry


class InferenceRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=1, reward_model=ExactMatchRewardModel()),
    ]


class BaseInferenceTask(BaseTextTask):
    query: str | None = None
    reference: str | None = None
    model: ModelConfig
    seed: int = random.randint(0, 1_000_000)

    @abstractmethod
    def make_query(self, dataset_entry: DatasetEntry) -> str:
        raise NotImplementedError("Method make_query must be implemented")

    def make_reference(self, dataset_entry: DatasetEntry) -> str:
        if self.model in model_manager.active_models.keys():
            self.reference = model_manager.active_models[self.model].generate(
                self.query, SamplingParams(seed=self.seed)
            )
            return self.reference


class OrganicInferenceData(BaseInferenceTask):
    seed: int = random.randint(0, 1_000_000)

    def make_query(self, dataset_entry: DatasetEntry) -> str:
        assert self.query is not None, "Organic Inference Tasks must be spawned with query"
        return self.query


class SyntheticInferenceTask(BaseInferenceTask):
    model: ModelConfig = ModelZoo.get_random()

    def make_query(self, dataset_entry: MMLUEntry) -> str:
        self.query = dataset_entry.query
        return self.query
