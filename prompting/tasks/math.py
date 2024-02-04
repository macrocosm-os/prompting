import sys
import bittensor as bt
from dataclasses import dataclass
from prompting.tasks import Task


@dataclass
class MathTask(Task):
    reward_definition = [
        dict(name='float_diff', weight = 1.0),
    ]
    penalty_definition = []

    def __init__(self, llm_pipeline, context, create_reference=True):
        
        reference = context["solution"]

        self.name="math"
        self.desc="get help solving a math problem"
        self.goal="to get the answer to the following math question"
        
        self.context = context

        query = "How can I solve, " + context["problem"] + "?"
        
        self.query=query
        self.reference=str(reference)
        self.topic=context["topic"]
        self.subtopic=context["subtopic"]
        self.tags=[]
        self.static_reference=True
        self.static_query=True