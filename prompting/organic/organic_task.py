from prompting.tasks.base_task import BaseTextTask
from prompting.rewards.reward import BaseRewardConfig, WeightedRewardModel
from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.utils.cleaners import CleanerPipeline
from prompting.llms.vllm_llm import vLLM_LLM
from prompting.llms.base_llm import BasePipeline
from prompting.settings import settings
from prompting.tasks.base_task import CHATTENSOR_SYSTEM_PROMPT
from typing import ClassVar, Any
from prompting.llms.model_manager import model_manager


class OrganicRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel()),
        WeightedRewardModel(weight=0.5, reward_model=RelevanceRewardModel()),
    ]
    penalty_definition: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel())
    ]


class OrganicTask(BaseTextTask):
    """Task with defined reward and penalty mechanisms for organic prompts."""

    cleaning_pipeline: ClassVar[CleanerPipeline] = CleanerPipeline()

    @classmethod
    async def generate_reference(cls, messages: list[str], roles: list[str], pipeline: BasePipeline) -> str:
        """Generate reference for the given organic or synthetic sample."""
        reference = vLLM_LLM(
            llm_pipeline=model_manager.get_model(settings.NEURON_MODEL_ID_VALIDATOR),
            system_prompt=CHATTENSOR_SYSTEM_PROMPT(),
            max_new_tokens=settings.ORGANIC_REFERENCE_MAX_TOKENS,
        ).query_conversation(
            messages=messages,
            roles=roles,
        )
        return cls.cleaning_pipeline.apply(reference)
