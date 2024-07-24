import bittensor as bt
from dataclasses import dataclass
from prompting.tasks import Task
from .challenge_templates import MathChallengeTemplate


@dataclass
class MathTask(Task):
    name: str = "math"
    desc: str = "get help solving a math problem"
    goal: str = "to get the answer to the following math question"

    reward_definition: list = [
        {"name": "float_diff", "weight": 0.8},
        {"name": "relevance", "weight": 0.2},
    ]
    penalty_definition: list = []

    static_reference: bool = True
    static_query: bool = True
    challenge_type: str = 'paraphrase'
    challenge_template: MathChallengeTemplate = MathChallengeTemplate()

    def __init__(self, llm_pipeline, context, create_reference: bool = True):
        self.context = context
        self.query = context.content
        self.reference = context.extra["solution"]
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags