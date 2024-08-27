from typing import ClassVar
from prompting.rewards.reward import WeightedRewardModel, BaseRewardConfig
from prompting.rewards.inference_reward_model import InferenceRewardModel
from loguru import logger

from prompting.tasks.base_task import BaseTextTask
from prompting.datasets.base import DatasetEntry
from prompting.llms.model_zoo import ModelConfig
import random
from prompting.llms.model_manager import model_manager
from vllm import RequestOutput, SamplingParams
from abc import abstractmethod
from prompting.datasets.mmlu import MMLUEntry


class InferenceRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=1, reward_model=InferenceRewardModel()),
    ]


class BaseInferenceTask(BaseTextTask):
    query: str | None = None
    reference: str | None = None
    model: ModelConfig | None = None
    seed: int = random.randint(0, 1_000_000)

    @abstractmethod
    def make_query(self, dataset_entry: DatasetEntry) -> str:
        raise NotImplementedError("Method make_query must be implemented")

    def make_reference(self, dataset_entry: DatasetEntry) -> str:
        if self.model is None:
            self.model = random.choice(list(model_manager.active_models.keys()))
        if self.model not in model_manager.active_models.keys():
            raise Exception(f"Model {self.model} not found in active models")
        output: RequestOutput = model_manager.active_models[self.model].generate(
            self.query, SamplingParams(seed=self.seed)
        )[0]
        self.reference = output.outputs[0].text
        if self.reference is None:
            logger.error(f"Model {self.model} returned None for reference generation")
        return self.reference


class OrganicInferenceTask(BaseInferenceTask):
    seed: int = random.randint(0, 1_000_000)

    def make_query(self, dataset_entry: DatasetEntry) -> str:
        assert self.query is not None, "Organic Inference Tasks must be spawned with query"
        return self.query


class SyntheticInferenceTask(BaseInferenceTask):
    # TODO: Once we want to enable the 'actual' inference task with exact models
    model: ModelConfig = None
    # this should be uncommented. For now, we're allowing non-exact responses, same
    # as the organic scoring task.
    # model: ModelConfig = ModelZoo.get_random()

    def make_query(self, dataset_entry: MMLUEntry) -> str:
        self.query = dataset_entry.query
        return self.query
