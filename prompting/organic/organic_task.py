from prompting.tasks.base_task import BaseTask
from prompting.rewards.reward import BaseRewardConfig, WeightedRewardModel
from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.utils.cleaners import CleanerPipeline
from prompting.llms.vllm_llm import vLLM_LLM
from prompting.llms.base_llm import BasePipeline
from prompting import settings
from prompting.tasks.base_task import CHATTENSOR_SYSTEM_PROMPT
from typing import ClassVar, Any


class OrganicRewardConfig(BaseRewardConfig):
    reward_definitions: list[WeightedRewardModel] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel()),
        WeightedRewardModel(weight=0.5, reward_model=RelevanceRewardModel()),
    ]
    penalty_definition: list[WeightedRewardModel] = [WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel())]


class OrganicTask(BaseTask):
    """Task with defined reward and penalty mechanisms for organic prompts."""

    cleaning_pipeline: ClassVar[CleanerPipeline] = CleanerPipeline()

    @classmethod
    async def generate_reference(cls, sample: dict[str, Any], pipeline: BasePipeline) -> str:
        """Generate reference for the given organic or synthetic sample."""
        reference = vLLM_LLM(
            llm_pipeline=pipeline,
            system_prompt=CHATTENSOR_SYSTEM_PROMPT(),
            max_new_tokens=settings.ORGANIC_REFERENCE_MAX_TOKENS,
        ).query_conversation(
            messages=sample["messages"],
            roles=sample["roles"],
        )
        return cls.cleaning_pipeline.apply(reference)
