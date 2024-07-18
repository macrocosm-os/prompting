from dataclasses import dataclass

from prompting.tasks import Task


@dataclass
class OrganicTask(Task):
    """Task with defined reward and penalty mechanisms for organic prompts."""
    name = "organic"
    # Use challenge as a query.
    challenge_type = "query"

    reward_definition = [
        # dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
        dict(name="relevance", weight=2.0),
    ]

    penalty_definition = [
        dict(name="relevance", weight=2.0),
    ]

    cleaning_pipeline = []

    def __init__(self, context: dict, reference: str):
        self.context = context
        self.roles = context["roles"]
        self.query = context["messages"][-1]
        self.reference = reference

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, query={self.query!r}, reference={self.reference!r})"

    def __repr__(self):
        return str(self)

    def __state_dict__(self, full=False):
        # Disable any logs for organic queries.
        state = {}
        return state
