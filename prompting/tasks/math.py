import sys
import bittensor as bt
from dataclasses import dataclass
from prompting.llms.base_llm import BasePipeline
from prompting.shared.context import Context
from prompting.tasks import Task
from .challenge_templates import MathChallengeTemplate


@dataclass
class MathTask(Task):
    name = "math"
    desc = "get help solving a math problem"
    goal = "to get the answer to the following math question"

    reward_definition = [
        dict(name="float_diff", weight=1.0),
    ]
    penalty_definition = []

    static_reference = True
    static_query = True
    challenge_type = "paraphrase"
    challenge_template = MathChallengeTemplate()

    def __init__(
        self,
        llm_pipeline: BasePipeline,
        context: Context,
        create_reference: bool = True,
    ):
        self.context: Context = context
        self.query: str = context.content
        self.reference: str = context.extra["solution"]
        self.topic: str = context.title
        self.subtopic: str = context.topic
        self.tags: list[str] = context.tags
