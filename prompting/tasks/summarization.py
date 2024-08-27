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
You are a request-generating expert. When asked to generate a request, you ask for a detailed summary of a topic. Your request should be specificly about the topic.
"""

REFERENCE_SYSTEM_PROMPT = """\
You are an expert question-answering LLM. You will receive context and a question, and you will generate a detailed and accurate answer to the question. Your answer should be based on the context provided.
"""

QUERY_PROMPT_TEMPLATE = """\
Request an exhaustive summary about the topic: {title}"""

# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\
Summarize the following context in a concise and accurate manner:

## Context
{context}
"""

class SummarizationRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel()),
        WeightedRewardModel(weight=0.5, reward_model=RelevanceRewardModel()),
    ]
    penalty_definition: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel())
    ]


class SummarizationTask(BaseTask):
    name: ClassVar[str] = "summarization"
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
        query = cls.generate_query(llm_pipeline=llm_pipeline, message=query_prompt)
        reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=context.content, question=query)
        reference = cls.generate_reference(llm_pipeline=llm_pipeline, messages=[reference_prompt])
        return query, reference
