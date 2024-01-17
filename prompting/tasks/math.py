import sys
import bittensor as bt
from dataclasses import dataclass
from prompting.tasks import Task


@dataclass
class MathTask(Task):
    reward_definition = [
        dict(name='float_diff', weight = 1.0),
    ]

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context
        query = "How can I solve, " + self.context["problem"] + "?"
        reference = self.context["solution"]
        
        try:
            float(reference)
        except:
            bt.logging.error(f"Solution {reference} is not a float.{sys.exc_info()}")
            raise ValueError(f"Solution {reference} is not a float.") 

        super().__init__(
            name="math",
            desc="get help solving a math problem",
            goal="to get the answer to the following math question",
            query=query,
            reference=str(reference),
            topic=self.context["problem"],
            subtopic="",
            tags="",
            static_reference=True,
            static_query=True,
        )
