from prompting.llms.base_llm import BasePipeline
from prompting.shared.context import Context
from prompting.tasks import Task
from .challenge_templates import SentimentChallengeTemplate

QUERY_PROMPT_TEMPLATE = """\
You are a review-generating expert, focusing on making highly reaslistic revies. Your response contains only the review, nothing more, nothing less. You will adhere to a word limit of 250 words. Ask a specific question about the following context:
{context}
"""


class SentimentAnalysisTask(Task):
    name = "sentiment"
    desc = "get help analyzing the sentiment of a review"
    goal = "to get the sentiment to the following review"
    challenge_type = "paraphrase"
    challenge_template = SentimentChallengeTemplate()

    reward_definition = [
        dict(name="ordinal", weight=1.0),
    ]
    penalty_definition = []
    cleaning_pipeline = []

    static_reference = True

    def __init__(
        self,
        llm_pipeline: BasePipeline,
        context: Context,
        create_reference: bool = True,
    ):
        self.context: Context = context
        self.query_prompt: str = QUERY_PROMPT_TEMPLATE.format(context=context.content)
        self.query: str = self.generate_query(llm_pipeline)
        self.reference: str = context.subtopic

        self.topic: str = context.title
        self.subtopic: str = context.topic
        self.tags: list[str] = context.tags

    def format_challenge(self, challenge) -> str:
        return challenge.format(context=self.query)
