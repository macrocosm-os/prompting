from typing import ClassVar

from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.date import DateRewardModel
from prompting.tasks.base_task import BaseTask
from prompting.llms.base_llm import BasePipeline
from prompting.utils.cleaners import RemoveTags, FirstQuestion, CleanerPipeline
from prompting.datasets.wiki import DateContext
from prompting.rewards.reward import BaseRewardConfig, WeightedRewardModel
from typing import ClassVar

from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.date import DateRewardModel
from prompting.tasks.base_task import BaseTask
from prompting.llms.base_llm import BasePipeline
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
        WeightedRewardModel(weight=0.7, reward_model=DateRewardModel()),
        WeightedRewardModel(weight=0.3, reward_model=RougeRewardModel()),
    ]


class DateQuestionAnsweringTask(BaseTask):
    name: ClassVar[str] = "date_qa"
    cleaner: ClassVar[CleanerPipeline] = CleanerPipeline(cleaning_pipeline=[RemoveTags(), FirstQuestion()])
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""

    @classmethod
    def generate_query_reference(cls, llm_pipeline: BasePipeline, context: DateContext):
        query_prompt = QUERY_PROMPT_TEMPLATE.format(content=context.date, topic=context.title) #TODO Sort out context dictionary
        query = cls.generate_query(llm_pipeline=llm_pipeline, message=query_prompt)

        reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(date=context.content, query=query, content=context.subtopic)
        reference = cls.generate_reference(llm_pipeline=llm_pipeline, messages=[reference_prompt])

        return query, reference
