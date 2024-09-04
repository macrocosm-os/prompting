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
from prompting.datasets.random_website import DDGDatasetEntry


class InferenceRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        InferenceRewardModel(),
    ]


class BaseInferenceTask(BaseTextTask):
    query: str | None = None
    reference: str | None = None
    llm_model: ModelConfig | None = None
    seed: int = random.randint(0, 1_000_000)

    @abstractmethod
    def make_query(self, dataset_entry: DatasetEntry) -> str:
        raise NotImplementedError("Method make_query must be implemented")

    def make_reference(self, dataset_entry: DatasetEntry) -> str:
        responses: RequestOutput = model_manager.generate(
            prompts=[self.query], model=self.llm_model, sampling_params=SamplingParams(seed=self.seed)
        )[0]
        self.reference = responses[0]
        if self.reference is None:
            logger.error(f"Model {self.llm_model} returned None for reference generation")
        return self.reference


class OrganicInferenceTask(BaseInferenceTask):
    seed: int = random.randint(0, 1_000_000)

    def make_query(self, dataset_entry: DatasetEntry) -> str:
        assert self.query is not None, "Organic Inference Tasks must be spawned with query"
        return self.query


QUERY_PROMPT = """
Ask a question about the following text:

{website_content}

---

Ask a question about the text and nothing else:"""


class SyntheticInferenceTask(BaseInferenceTask):
    # TODO: Once we want to enable the 'actual' inference task with exact models
    model: ModelConfig = None

    def make_query(self, dataset_entry: DDGDatasetEntry) -> str:
        website_content = dataset_entry.website_content
        self.query = model_manager.generate(
            prompts=QUERY_PROMPT.format(website_content=website_content), model=self.llm_model
        )[0]
        return self.query
