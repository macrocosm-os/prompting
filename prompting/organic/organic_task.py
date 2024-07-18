from dataclasses import dataclass

from prompting.tasks import Task


@dataclass
class OrganicTask(Task):
    """Task with defined reward and penalty mechanisms for organic prompts."""
    name = "organic"
    # Use challenge as a query.
    challenge_type = "query"

    reward_definition = [dict(name="relevance", weight=1.0)]

    penalty_definition = [dict(name="relevance", weight=1.0)]

    cleaning_pipeline = []

    def __init__(self, context: dict, reference: str):
        self.context = context
        self.messages = context["messages"]
        self.roles = context["roles"]
        self.query = context["messages"][-1]
        self.topic = "Organic"
        self.reference = reference
        self.subtopic = ""
        self.tags = [""]

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, query={self.query!r}, reference={self.reference!r})"

    def __repr__(self):
        return str(self)

    def __state_dict__(self, full=False):
        # Disable any logs for organic queries.
        state = {}
        return state
