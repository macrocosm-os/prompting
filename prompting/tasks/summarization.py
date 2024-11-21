from prompting.tasks.base_task import BaseTextTask
from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BaseRewardModel
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


def make_query_prompt(context: Context) -> str:
    return "Creatively ask for a summary of the following context:\n\n" + context.title


class SummarizationRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        RougeRewardModel(weight=0.5),
        RelevanceRewardModel(weight=0.5),
    ]
    penalty_definition: ClassVar[list[BaseRewardModel]] = [RougeRewardModel(weight=0.5)]


class SummarizationTask(BaseTextTask):
    cleaning_pipeline: ClassVar[CleanerPipeline] = CleanerPipeline(
        cleaning_pipeline=[
            RemoveQuotes(),
            PruneEnding(),
            RemoveRoles(),
        ]
    )
    name: ClassVar[str] = "summarization"
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    reference_system_prompt: ClassVar[str] = REFERENCE_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""
    query: str | None = None
    reference: str | None = None

    def make_query(self, dataset_entry: Context):
        query_prompt = QUERY_PROMPT_TEMPLATE.format(title=dataset_entry.title)
        self.query = self.generate_query(messages=[query_prompt])
        return self.query

    def make_reference(self, dataset_entry: Context):
        reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=dataset_entry.content, question=self.query)
        self.reference = self.generate_reference(messages=[{"role": "user", "content": reference_prompt}])
        return self.reference
