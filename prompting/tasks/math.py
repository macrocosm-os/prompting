import sys
import bittensor as bt
from dataclasses import dataclass
from prompting.tasks import Task


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

    def __init__(self, llm_pipeline, context, create_reference=True):

        self.context = context

        self.query = (
            "How can I solve the following problem, "
            + context.content
            + "? Make sure to include the whole problem when you ask your question."
        )
        self.reference = context.extra["solution"]
        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
