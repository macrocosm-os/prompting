from prompting.tasks.task import Task
from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.tasks.task import WeightedRewardModel
from prompting.tasks.task import BaseRewardConfig
from prompting.cleaners.all_cleaners import RemoveRoles, RemoveQuotes, PruneEnding
from prompting.cleaners.cleaner import BaseCleaner
from prompting.shared import Context
from pydantic import model_validator
from prompting.llms.base_llm import BasePipeline

# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# TODO: Also add a query system prompt and a query prompt template
# TODO: Add the option to generate the summary query from the context. e.g. "the childhood of Abraham Lincoln" which is more specific than summarizing the entire article (Abraham Lincoln)

# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\
Summarize the following context in a concise and accurate manner:

## Context
{context}
"""


class SummarizationRewardConfig(BaseRewardConfig):
    reward_definitions: list[WeightedRewardModel] = [
        WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel()),
        WeightedRewardModel(weight=0.5, reward_model=RelevanceRewardModel()),
    ]
    penalty_definition: list[WeightedRewardModel] = [WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel())]


class SummarizationTask(Task):
    context: Context
    llm_pipeline: BasePipeline

    reference_prompt: str | None = None
    query: str | None = None
    create_reference: bool = True

    cleaning_pipeline: list[BaseCleaner] = [
        RemoveQuotes,
        PruneEnding,
        RemoveRoles,
    ]
    static_query: bool = True

    @model_validator(mode="after")
    def create_query_reference_prompt(self) -> "SummarizationTask":
        self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=self.context.content)
        if self.create_reference:
            self.reference = self.generate_reference(self.llm_pipeline)
        return self
