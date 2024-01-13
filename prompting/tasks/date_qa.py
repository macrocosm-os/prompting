import bittensor as bt
from dataclasses import dataclass
from prompting.tasks import Task
import textwrap


@dataclass
class DateQuestionAnsweringTask(Task):
    reward_definition = [
        dict(name="rouge", ngram="rouge-l", metric="f", weight=1.0),
    ]

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context
        section = self.context["section"]
        year, _, *event = self.context["event"].split()
        self.context["event"] = " ".join(event)
        if section == "events":
            query = "what date did " + self.context["event"] + " take place?"
        elif section == "births":
            query = "when was " + self.context["event"] + " born?"
        elif section == "deaths":
            query = "when did " + self.context["event"] + " die?"
        reference = self.context["date"] + ", " + year.strip()
        super().__init__(
            name="date-based question answering",
            desc="get help answering a question",
            goal=f"to get the answer to the following question about dates",
            query=query,
            reference=reference,
            topic=self.context["event"],
            subtopic="",
            tags="",
            static_reference=True,
            static_query=True,
        )
