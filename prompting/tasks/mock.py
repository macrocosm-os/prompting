from dataclasses import dataclass
from prompting.llms.base_llm import BasePipeline
from prompting.shared.context import Context
from prompting.tasks import Task


@dataclass
class MockTask(Task):
    name = "mock"
    desc = "get help solving a math problem"
    goal = "to get the answer to the following math question"

    reward_definition = [
        dict(name="float_diff", weight=1.0),
    ]
    penalty_definition = []

    static_reference = True
    static_query = True

    def __init__(
        self,
        llm_pipeline: BasePipeline,
        context: Context,
        create_reference: bool = True,
    ):
        self.context: Context = context

        self.query: str = (
            "How can I solve the following problem, " + context.content + "?"
        )
        self.reference: str = "This is the reference answer"
        self.topic: str = context.title
        self.subtopic: str = context.topic
        self.tags: list[str] = context.tags
