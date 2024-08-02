# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# TODO: Also add a query system prompt and a query prompt template
# TODO: Add the option to generate the summary query from the context. e.g. "the childhood of Abraham Lincoln" which is more specific than summarizing the entire article (Abraham Lincoln)

from prompting.tasks.base_task import BaseTask
from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import WeightedRewardModel
from prompting.rewards.reward import BaseRewardConfig
from prompting.utils.cleaners import RemoveRoles, RemoveQuotes, PruneEnding
from prompting.datasets.base import Context
from prompting.utils.cleaners import CleanerPipeline
from typing import ClassVar

QUERY_SYSTEM_PROMPT = """\
You are a question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity. The questions you generate should be based on the context that is provided.
You will maintain a neutral tone in your questions.
You will adhere to a word limit of 50 words for each question.
"""

REFERENCE_SYSTEM_PROMPT = """\
You are an expert question-answering LLM. You will receive context and a question, and you will generate a detailed and accurate answer to the question. Your answer should be based on the context provided.
"""

QUERY_PROMPT_TEMPLATE = """\
    Provide an exhaustive summary about the topic \"{title}\""""
# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\
Summarize the following context in a concise and accurate manner:

## Context
{context}
"""


def make_query_prompt(context: Context) -> str:
    return "Creatively ask for a summary of the following context:\n\n" + context.title


class SummarizationRewardConfig(BaseRewardConfig):
    reward_definitions: list[WeightedRewardModel] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel()),
        WeightedRewardModel(weight=0.5, reward_model=RelevanceRewardModel()),
    ]
    penalty_definition = [dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5)]

    # This is where you define cleaning procedures for the generation.
    # Can be used when wanting to clean the challenge.
    cleaning_pipeline = [
        dict(name="remove_quotes"),
        dict(name="prune_ending"),
        dict(name="remove_roles"),
    ]

    static_query = True

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        # Query is just the article title and section name
        self.query = context.title

        self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=context.content)
        if create_reference:
            self.reference = self.generate_reference(llm_pipeline)

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags

    penalty_definition: list[WeightedRewardModel] = [WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel())]


class SummarizationTask(BaseTask):
    cleaning_pipeline: ClassVar[CleanerPipeline] = CleanerPipeline(
        cleaning_pipeline=[
            RemoveQuotes(),
            PruneEnding(),
            RemoveRoles(),
        ]
    )
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    reference_system_prompt: ClassVar[str] = REFERENCE_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""

    @classmethod
    def generate_query_reference(cls, llm_pipeline, context: Context):
        query_prompt = QUERY_PROMPT_TEMPLATE.format(title=context.title)
        query = cls.generate_query(llm_pipeline=llm_pipeline, messages=[query_prompt])
        reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=context.content, question=query)
        reference = cls.generate_reference(llm_pipeline=llm_pipeline, messages=[reference_prompt])
        return query, reference
