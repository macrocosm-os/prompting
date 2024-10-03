import textwrap
from typing import ClassVar, Optional

from prompting.datasets.random_website import DDGDatasetEntry
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
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
"""Ask a question about the following text in such a way that it's not obvious 
that you're asking about text from this specific website, but keep the context to make sure that the 
question can be answered through the internet search.
"""
)
QUERY_PROMPT_TEMPLATE = "[Input Text]\n{context}"


class WebRetrievalRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        WebRetrievalRewardModel(weight=1.0),
    ]


class WebRetrievalTask(BaseTextTask):
    name: ClassVar[str] = "web_retrieval"
    augmentation_system_prompt: ClassVar[str] = ""
    query_system_prompt: ClassVar[Optional[str]] = QUERY_SYSTEM_PROMPT

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
        self.reference = dataset_entry.model_dump_json()
        return self.reference
