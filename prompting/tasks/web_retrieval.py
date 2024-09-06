import json
from typing import ClassVar, Optional

import numpy as np
import textwrap

from prompting.datasets.base import Context
from prompting.datasets.random_website import DDGDatasetEntry
from prompting.rewards.reward import BaseRewardConfig, WeightedRewardModel
from prompting.rewards.web_retrieval import WebRetrievalRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.tasks.base_task import BaseTask, BaseTextTask
from prompting.utils.exceptions import TaskCreationError


# Used to instruct the LLM to provide a query when given a context.
QUERY_SYSTEM_PROMPT = textwrap.dedent("""You're an LLM agent that helps users develop better research skills. 
Ask a question about the following text in such a way that it's not obvious 
that you're asking about text from this specific website. Make it such that the 
question can be answered by doing a thorough search on the internet.
"""
)

QUERY_PROMPT_TEMPLATE = "[Input Text]\n{context}"


class WebRetrievalRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        # WeightedRewardModel(weight=1.0, reward_model=WebRetrievalRewardModel()),
        WeightedRewardModel(weight=1.0, reward_model=RelevanceRewardModel()),
    ]


class WebRetrievalTask(BaseTextTask):
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""
    # llm_model_id: Optional[str] = None
    query_system_prompt: ClassVar[str | None] = QUERY_SYSTEM_PROMPT

    def make_query(self, context: DDGDatasetEntry) -> str:
        query_prompt = QUERY_PROMPT_TEMPLATE.format(context=context.website_content)
        query = self.generate_query(messages=query_prompt)
        return query

    def make_reference(self, context: Context) -> str:
        return self.reference
