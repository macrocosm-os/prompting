from typing import ClassVar

from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.date import DateRewardModel
from prompting.tasks.base_task import BaseTextTask
from prompting.utils.cleaners import RemoveTags, FirstQuestion, CleanerPipeline
from prompting.datasets.wiki import DateContext
from prompting.rewards.reward import BaseRewardConfig, WeightedRewardModel

QUERY_SYSTEM_PROMPT = """You are a question creation expert. When asked to create a question, you use the context to make a specific question that would have the answer <date>. Your question should contain the topic."""
QUERY_PROMPT_TEMPLATE = """\
Create a question about {topic} that would have <date> as the answer using the following context:
context: {content}
"""
REFERENCE_PROMPT_TEMPLATE = """\
Your answer must include the following date: {date}.
Answer the following question using the provided context.
Question: {query}
Context: {content}
"""


class DateQARewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        DateRewardModel(weight=0.7),
        RougeRewardModel(weight=0.3),
    ]


class DateQuestionAnsweringTask(BaseTextTask):
    name: ClassVar[str] = "date_qa"
    cleaner: ClassVar[CleanerPipeline] = CleanerPipeline(cleaning_pipeline=[RemoveTags(), FirstQuestion()])
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""
    query: str | None = None
    reference: str | None = None

    def make_query(self, dataset_entry: DateContext, **kwargs) -> str:
        query_prompt = QUERY_PROMPT_TEMPLATE.format(content=dataset_entry.content, topic=dataset_entry.topic)
        self.query = self.generate_query(messages=[query_prompt])
        return self.query

    def make_reference(self, dataset_entry: DateContext) -> str:
        reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(
            date=dataset_entry.date, query=self.query, content=dataset_entry.content
        )
        self.reference = self.generate_reference(messages=[reference_prompt])
        return self.reference
