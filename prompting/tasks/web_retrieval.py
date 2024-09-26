import textwrap
from typing import ClassVar, Optional

from prompting.datasets.random_website import DDGDatasetEntry
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BaseRewardConfig, WeightedRewardModel
from prompting.tasks.base_task import BaseTextTask

# Used to instruct the LLM to provide a query when given a context.
QUERY_SYSTEM_PROMPT = textwrap.dedent(
    """You're an LLM agent that helps users develop better research skills.
Ask a question about the following text in such a way that it's not obvious
that you're asking about text from this specific website. Make it such that the
question can be answered by doing a thorough search on the internet.
"""
)

QUERY_PROMPT_TEMPLATE = "[Input Text]\n{context}"


class WebRetrievalRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        RelevanceRewardModel(weight=1.0),
    ]


class WebRetrievalTask(BaseTextTask):
    name: ClassVar[str] = "web_retrieval"
    augmentation_system_prompt: ClassVar[str] = ""
    query_system_prompt: ClassVar[Optional[str]] = QUERY_SYSTEM_PROMPT

    def make_query(self, dataset_entry: DDGDatasetEntry) -> str:
        query_prompt = QUERY_PROMPT_TEMPLATE.format(context=dataset_entry.website_content)
        self.query = self.generate_query(messages=query_prompt)
        return self.query

    def make_reference(self, dataset_entry: DDGDatasetEntry) -> str:
        self.reference = dataset_entry.website_content
        return self.reference
