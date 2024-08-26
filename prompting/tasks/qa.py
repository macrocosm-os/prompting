from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.tasks.base_task import BaseTask
from prompting.rewards.reward import WeightedRewardModel

# from prompting.rewards.reward import BaseRewardModel
from prompting.utils.cleaners import RemoveRoles, RemoveQuotes, PruneEnding, RemovePostQuestionText
from prompting.utils.cleaners import CleanerPipeline
from prompting.datasets.base import Context
from prompting.rewards.reward import BaseRewardConfig
from typing import ClassVar

# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# Used to instruct the LLM to provide a good query when given a context
QUERY_SYSTEM_PROMPT = """\
You are a question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity. The questions you generate should be based on the context that is provided.
You will maintain a neutral tone in your questions.
You will adhere to a word limit of 50 words for each question.
"""

REFERENCE_SYSTEM_PROMPT = """\
You are an expert question-answering LLM. You will receive context and a question, and you will generate a detailed and accurate answer to the question. Your answer should be based on the context provided.
"""

# Used to obtain the query (which is a question about the context)
QUERY_PROMPT_TEMPLATE = """\
Ask a specific question about the following context:

#Context:
{context}

You must ask a question that can be answered by the context.
"""


# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\
Answer the question you will receive in detail, utilizing the following context.

#Context:
{context}

# Question:
{question}
"""

class QARewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel()),
        WeightedRewardModel(weight=0.5, reward_model=RelevanceRewardModel()),
    ]
    penalty_definition: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel())
    ]


class QuestionAnsweringTask(BaseTask):
    """QuestionAnsweringTasks must be initialised with an LLM pipeline to generate query and reference plus
    context from a dataset to base the query on"""
    name: ClassVar[str] = "qa"
    cleaning_pipeline: ClassVar[CleanerPipeline] = CleanerPipeline(
        cleaning_pipeline=[
            RemoveQuotes(),
            PruneEnding(),
            RemoveRoles(),
            RemovePostQuestionText(),
        ]
    )
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    reference_system_prompt: ClassVar[str] = REFERENCE_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""

    @classmethod
    def generate_query_reference(cls, llm_pipeline, context: Context):
        query_prompt = QUERY_PROMPT_TEMPLATE.format(context=context.content)
        query = cls.generate_query(llm_pipeline=llm_pipeline, message=query_prompt)
        reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=context.content, question=query)
        reference = cls.generate_reference(llm_pipeline=llm_pipeline, messages=[reference_prompt])
        return query, reference
