from dataclasses import dataclass

from prompting.organic.organic_task import OrganicTask


@dataclass
class SynthOrganicTask(OrganicTask):
    """Task with defined reward and penalty mechanisms for synthetic organic prompts."""
    name = "synthetic-organic"

    reward_definition = [
        # dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
        dict(name="relevance", weight=1.0),
    ]
    penalty_definition = [
        dict(name="relevance", weight=1.0),
    ]
