import random
import json
from typing import ClassVar

from loguru import logger

from prompting.datasets.msr_v2_dataset import MSRDiscrimintaorDatasetEntry
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from prompting.rewards.discriminator import DiscriminatorRewardModel
from prompting.tasks.multi_step_reasoning import MultiStepReasoningTask, MultiStepReasoningRewardConfig
from shared.base import Context
from validator_api.test_time_inference import generate_response

MAX_THINKING_STEPS = 10


def execute_multi_step_reasoning(user_query: str):
    for steps, total_thinking_time in generate_response(user_query):
        if total_thinking_time is not None:
            logger.info(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
    return steps, total_thinking_time


class MultiStepReasoningGeneratorRewardConfig(MultiStepReasoningRewardConfig):
    """The reward config for the generator task is the same as for the normal msr task"""
    pass

class MultiStepReasoningDiscriminatorRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        DiscriminatorRewardModel(weight=1),
    ]




class MultiStepReasoningTaskGenerator(MultiStepReasoningTask):
    """QuestionAnsweringTasks must be initialised with an LLM pipeline to generate query and reference plus
    context from a dataset to base the query on"""

    name: ClassVar[str] = "multi_step_reasoning_v2"

class MultiStepReasoningTaskDiscriminator(MultiStepReasoningTask):
    name: ClassVar[str] = "multi_step_reasoning_discriminator"
    augmentation_system_prompt: ClassVar[str] = ""
    query: str | None = None
    query_system_prompt: str = QUERY_SYSTEM_PROMPT
    reference: str | None = None
    original_reference: str | None = None
    miner_response: str | None = None
    correct_answer: str | None = None
    original_miner_uid: int | None = None

    def make_query(self, dataset_entry: MSRDiscrimintaorDatasetEntry):
        options = [self.original_reference, self.miner_response]
        random.shuffle(options)
        option_a, option_b = options
        
        # Track which option is the correct answer
        self.correct_answer = "A" if option_a == self.original_reference else "B"
        self.original_miner_uid = dataset_entry.miner_uid
        
        json_question = {
            "option_a": option_a,
            "option_b": option_b,
        }
        self.messages = [{"role": "user", "content": json.dumps(json_question)}]
        self.query = self.messages[-1]["content"]
        return self.query

    async def make_reference(self, dataset_entry: Context):
        self.reference = self.correct_answer
        return self.reference


