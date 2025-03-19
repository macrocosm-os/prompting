import json
import random
from typing import ClassVar

from loguru import logger

from prompting.datasets.msr_v2_dataset import MSRDiscriminatorDatasetEntry
from prompting.tasks.multi_step_reasoning import MultiStepReasoningTask
from shared.base import Context
from validator_api.test_time_inference import generate_response

MAX_THINKING_STEPS = 10


def execute_multi_step_reasoning(user_query: str):
    for steps, total_thinking_time in generate_response(user_query):
        if total_thinking_time is not None:
            logger.info(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
    return steps, total_thinking_time


class MultiStepReasoningTaskGenerator(MultiStepReasoningTask):
    """QuestionAnsweringTasks must be initialised with an LLM pipeline to generate query and reference plus
    context from a dataset to base the query on"""

    name: ClassVar[str] = "MultiStepReasoningTaskGenerator"


class MultiStepReasoningTaskDiscriminator(MultiStepReasoningTask):
    name: ClassVar[str] = "multi_step_reasoning_discriminator"
    augmentation_system_prompt: ClassVar[str] = ""
    query: str | None = None
    reference: str | None = None
    original_reference: str | None = None
    miner_response: str | None = None
    correct_answer: str | None = None
    original_miner_uid: int | None = None
    generator_task_id: str | None = None  # Track which Generator created this Discriminator

    def __init__(self, dataset_entry: MSRDiscriminatorDatasetEntry, generator_task_id: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_entry = dataset_entry
        self.generator_task_id = generator_task_id
        self.make_query(dataset_entry)

    def make_query(self, dataset_entry: MSRDiscriminatorDatasetEntry) -> str:
        """Creates a query by randomly shuffling the validator reference and miner response.

        Args:
            dataset_entry: Dataset entry containing validator reference and miner response

        Returns:
            JSON string containing the shuffled options as a query

        Raises:
            ValueError: If dataset entry is missing required fields
        """
        # Validate input
        if not dataset_entry.validator_reference:
            logger.error(f"Dataset entry missing required validator_reference: {dataset_entry.validator_reference}")
        if not dataset_entry.miner_response:
            logger.error(f"Dataset entry missing required miner_response: {dataset_entry.miner_response}")

        # Store the original values
        self.original_reference = dataset_entry.validator_reference
        self.miner_response = dataset_entry.miner_response
        self.original_miner_uid = dataset_entry.miner_uid

        # Create and shuffle options
        options = [self.original_reference, self.miner_response]
        random.shuffle(options)
        option_a, option_b = options

        # Track which option is the correct answer
        self.correct_answer = "A" if option_a == self.original_reference else "B"

        # Create query message
        query_content = {
            "option_a": option_a,
            "option_b": option_b,
        }

        try:
            query_json = json.dumps(query_content)
            self.messages = [{"role": "user", "content": query_json}]
            self.query = self.messages[-1]["content"]
            return self.query
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to create Descriminator JSON query: {str(e)}")

    async def make_reference(self, dataset_entry: Context):
        self.reference = self.correct_answer
        return self.reference
