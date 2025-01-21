import random
from typing import ClassVar

import numpy as np
from pydantic import Field, model_validator

from prompting.datasets.sn13 import ChatEntry
from prompting.llms.model_manager import model_manager
from prompting.llms.model_zoo import ModelConfig, ModelZoo
from prompting.rewards.inference_reward_model import InferenceRewardModel
from prompting.rewards.penalty import PenaltyModel
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from prompting.tasks.base_task import BaseTextTask
from shared.settings import shared_settings


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

SYSTEM_PROMPTS = [
    "",
    "You are a helpful AI assistant. Provide concise, accurate answers to any questions asked.",
    "You are a friendly and patient assistant. Communicate your responses in a clear, easy-to-understand way, ensuring the user feels supported.",
    "You are a creative helper. Offer engaging, imaginative responses that keep the user interested, while maintaining accuracy and clarity.",
]


class InferenceTask(BaseTextTask):
    name: ClassVar[str] = "inference"
    # TODO: Once we want to enable the 'actual' inference task with exact models
    query: str | None = None
    reference: str | None = None
    system_prompt: str | None = None
    llm_model: ModelConfig | None = None
    llm_model_id: ModelConfig | None = random.choice(ModelZoo.models_configs).llm_model_id
    seed: int = Field(default_factory=lambda: random.randint(0, 1_000_000), allow_mutation=False)
    sampling_params: dict[str, float] = shared_settings.SAMPLING_PARAMS.copy()

    @model_validator(mode="after")
    def random_llm_model_id(self):
        if self.query:  # If we are already defining query, as in the case of organics, we also specify model.
            return self
        # Choose system prompt and randomize inference settings
        self.system_prompt = random.choice(SYSTEM_PROMPTS)
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self.sampling_params["temperature"] = random.randint(0, 10) / 10
        self.sampling_params["max_new_tokens"] = random.choice([256, 512, 1024, 2048])

        if np.random.rand() < 0.2:
            self.llm_model_id = None
        else:
            self.llm_model = ModelZoo.get_model_by_id(self.llm_model_id)
        return self

    def make_query(self, dataset_entry: ChatEntry) -> str:
        if self.query:
            return self.query
        self.messages.extend(dataset_entry.messages)
        self.query = self.messages

        return self.query

    def make_reference(self, dataset_entry: ChatEntry) -> str:
        self.reference = model_manager.generate(
            messages=self.messages,
            model=self.llm_model,
            seed=self.seed,
            sampling_params=self.sampling_params,
        )
        return self.reference
