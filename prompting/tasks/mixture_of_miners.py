from typing import ClassVar

from vllm import SamplingParams

from prompting import mutable_globals
from prompting.datasets.mixture_of_miners import MixtureOfMinersEntry
from prompting.tasks.base_task import BaseTextTask
from prompting.llms.model_manager import model_manager
from prompting.tasks.inference import InferenceRewardConfig, InferenceTask
from prompting.tasks.web_retrieval import WebRetrievalTask


class MixtureOfMinersRewardConfig(InferenceRewardConfig):
    pass


class SystemPromptAggregator:
    def __init__(self):
        self.default_prompt = """You have been provided with a set of responses from various open-source models to the latest user query.
Your task is to synthesize these responses into a single, high-quality and concise response.
It is crucial to follow the provided instructions or examples in the given prompt if any, and ensure the answer is in correct and expected format.
Critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.
Your response should not simply replicate the given answers but should offer a refined and accurate reply to the instruction.
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

        self.task_prompts = {
            # TODO: Define web retrieval system prompt.
            WebRetrievalTask: None
        }

    def aggregate(self, task: BaseTextTask, completions: list[str]) -> str:
        """Construct the system prompt for the given task and aggregates the responses.

        Args:
            task_name (str): Primary task to be aggregated.
            completions: A dictionary with UIDs as keys and responses as values.

        Returns:
            str: The aggregated system prompt.

        Raises:
            NotImplementedError: If the task system prompt is not implemented.
        """
        # Check if default system prompt can be used.
        if task in self.task_prompts and self.task_prompts[task] is None:
            raise NotImplementedError(f"System prompt for '{task}' is not implemented.")

        # Get the task-specific prompt or use the default prompt.
        system_template = self.task_prompts.get(task, self.default_prompt)

        combined_completions = "\n".join([f"{i+1}. {comp}" for i, comp in enumerate(completions)])
        system_agg = f"{system_template}\n{combined_completions}"
        return system_agg


class MixtureOfMinersTask(InferenceTask):
    # name: ClassVar[str] = MixtureOfMinersTask.__name__
    # query: str | None = None
    # reference: str | None = None
    # llm_model: ModelConfig | None = None
    # llm_model_id: ModelConfig | None = random.choice(ModelZoo.models_configs).llm_model_id
    # seed: int = Field(default_factory=lambda: random.randint(0, 1_000_000))

    use_cached_tasks: ClassVar[bool] = False
    primary_task: BaseTextTask = False
    system_aggregator: ClassVar[SystemPromptAggregator] = SystemPromptAggregator()

    @classmethod
    def is_available(cls) -> bool:
        if cls.use_cached_tasks and not mutable_globals.task_responses:
            return False
        return True

    def make_query(self, dataset_entry: MixtureOfMinersEntry) -> str:
        if self.query:
            return self.query
        self.query = dataset_entry.messages[-1]
        self.messages = dataset_entry.messages
        self.synapse_system_prompt = self.system_aggregator.aggregate(
            primary_task=dataset_entry.primary_task,
            completions=dataset_entry.completions,
        )
        return self.query

    def make_reference(self, dataset_entry: MixtureOfMinersEntry) -> str:
        self.reference = model_manager.generate(
            messages=[self.synapse_system_prompt, self.messages[-1]],
            roles=["system", "user"],
            model=self.llm_model,
            sampling_params=SamplingParams(seed=self.seed),
        )[0]
        return self.reference
