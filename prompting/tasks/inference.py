from typing import ClassVar
from prompting.rewards.reward import BaseRewardModel, BaseRewardConfig
from prompting.rewards.inference_reward_model import InferenceRewardModel
from prompting.rewards.penalty import PenaltyModel

from prompting.tasks.base_task import BaseTextTask
from prompting.llms.model_zoo import ModelConfig
import random
from prompting.llms.model_manager import model_manager
from prompting.datasets.lmsys import ChatEntry


class InferenceRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        InferenceRewardModel(weight=0.5),
        PenaltyModel(weight=0.5),
    ]


QUERY_PROMPT = """
Ask a question about the following text:

{website_content}

---

Ask a question about the text and nothing else:"""


class InferenceTask(BaseTextTask):
    name: ClassVar[str] = "inference"
    # TODO: Once we want to enable the 'actual' inference task with exact models
    query: str | None = None
    reference: str | None = None
    llm_model: ModelConfig | None = None
    seed: int = random.randint(0, 1_000_000)

    def make_query(self, dataset_entry: ChatEntry) -> str:
        if self.query:
            return self.query
        self.query = dataset_entry.messages[-1]
        self.messages = dataset_entry.messages
        return self.query

    def make_reference(self, dataset_entry: ChatEntry) -> str:
        self.reference = model_manager.chat_generate(
            messages=self.messages, roles=dataset_entry.roles, model=self.llm_model
        )[0]
        return self.reference
