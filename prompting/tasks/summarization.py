# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# TODO: Also add a query system prompt and a query prompt template
# TODO: Add the option to generate the summary query from the context. e.g. "the childhood of Abraham Lincoln" which is more specific than summarizing the entire article (Abraham Lincoln)

from prompting.tasks.base_task import BaseTextTask
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
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel()),
        WeightedRewardModel(weight=0.5, reward_model=RelevanceRewardModel()),
    ]
    penalty_definition: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel())
    ]


class SummarizationTask(BaseTextTask):
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
    query: str | None = None
    reference: str | None = None

    def make_query(self, llm_pipeline, context: Context):
        query_prompt = QUERY_PROMPT_TEMPLATE.format(title=context.title)
        self.query = self.generate_query(llm_pipeline=llm_pipeline, messages=[query_prompt])
        return self.query

    def make_reference(self, llm_pipeline, context: Context):
        reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=context.content, question=self.query)
        self.reference = self.generate_reference(llm_pipeline=llm_pipeline, messages=[reference_prompt])
        return self.reference
