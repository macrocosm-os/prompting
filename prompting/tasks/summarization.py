from prompting.tasks.base_task import BaseTask
from prompting.rewards.rouge import RougeRewardModel
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import WeightedRewardModel
from prompting.rewards.reward import BaseRewardConfig
from prompting.utils.cleaners import RemoveRoles, RemoveQuotes, PruneEnding
from prompting.datasets.base import Context
from pydantic import model_validator
from prompting.utils.cleaners import CleanerPipeline

# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# TODO: Also add a query system prompt and a query prompt template
# TODO: Add the option to generate the summary query from the context. e.g. "the childhood of Abraham Lincoln" which is more specific than summarizing the entire article (Abraham Lincoln)

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
    penalty_definition: list[WeightedRewardModel] = [WeightedRewardModel(weight=0.5, reward_model=RougeRewardModel())]


class SummarizationTask(BaseTask):
    context: Context | None = None
    reference_prompt: str | None = None
    query: str | None = None
    create_reference: bool = True

    cleaning_pipeline: CleanerPipeline = CleanerPipeline(
        cleaning_pipeline=[
            RemoveQuotes(),
            PruneEnding(),
            RemoveRoles(),
        ]
    )

    @model_validator(mode="after")
    def create_query_reference_prompt(self) -> "SummarizationTask":
        if not self.query or not self.reference:
            assert self.context, "You must either initialise the task with context or both query and reference"
            self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(context=self.context.content)
            self.query_prompt = make_query_prompt(self.context)
