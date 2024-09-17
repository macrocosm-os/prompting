import textwrap
from typing import ClassVar, Optional

from prompting.datasets.random_website import DDGDatasetEntry
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BaseRewardConfig, WeightedRewardModel
from prompting.rewards.web_retrieval import WebRetrievalRewardModel
from prompting.tasks.base_task import BaseTextTask


MINER_EXAMPLE_1_SHOT = """\
[Example 1]
What is the capital of Texas?

Austin is the capital of the U.S. state of Texas and the seat and most populous city of Travis County, \
with portions extending into Hays and Williamson counties.

URL: https://en.wikipedia.org/wiki/Austin,_Texas
"""

# Used to instruct the LLM to provide a query when given a context.
QUERY_SYSTEM_PROMPT = textwrap.dedent(
"""Ask a question about the following text by keeping the context of the topic.
Make it such that the question can be answered by doing a thorough search on the internet.
"""
)
QUERY_PROMPT_TEMPLATE = "[Input Text]\n{context}"

# REFERENCE_SYSTEM_PROMPT = textwrap.dedent(
# """Your task is to answer the following question with the given context.
# """
# )
# REFERENCE_PROMPT_TEMPLATE = "[Question]\n{question}\n[Context]\n{context}"


class WebRetrievalRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        # WeightedRewardModel(weight=1.0, reward_model=RelevanceRewardModel()),
        WeightedRewardModel(weight=1.0, reward_model=WebRetrievalRewardModel()),
    ]


class WebRetrievalTask(BaseTextTask):
    name: ClassVar[str] = "web_retrieval"
    augmentation_system_prompt: ClassVar[str] = ""
    query_system_prompt: ClassVar[Optional[str]] = QUERY_SYSTEM_PROMPT
    # reference_system_prompt: ClassVar[Optional[str]] = REFERENCE_SYSTEM_PROMPT

    def make_query(self, dataset_entry: DDGDatasetEntry) -> str:
        query_prompt = QUERY_PROMPT_TEMPLATE.format(context=dataset_entry.website_content)
        question = self.generate_query(messages=query_prompt)
        prompt: list[str] = []
        prompt.append("Search the web for the given query, provide the content of the website and the URL.\n\n")
        prompt.append(f"{MINER_EXAMPLE_1_SHOT}\n")
        prompt.append(f"[Input Query]\n{question}\n")
        self.query = "".join(prompt)
        return self.query

    def make_reference(self, dataset_entry: DDGDatasetEntry) -> str:
        # Approach #1: Q&A.
        # reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=dataset_entry.website_content)
        # self.reference = self.generate_reference(reference_prompt)

        # Approach #2: Reference content and response content similarity.
        # self.reference = dataset_entry.website_content

        # Approach #3: Search term and response content similarity.
        # self.reference = dataset_entry.search_term
        self.reference = dataset_entry
        return self.reference
