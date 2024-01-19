from dataclasses import dataclass
from prompting.tasks import Task
from prompting.cleaners.cleaner import CleanerPipeline


@dataclass
class DateQuestionAnsweringTask(Task):
    reward_definition = [
        dict(name="date", weight=1),
    ]

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.cleaning_pipeline = [
            dict(name="remove_quotes"),
            dict(name="remove_roles"),
        ]

        self.context = context
        self.section = self.context["section"]
        year, _, *event = self.context["event"].split()
        self.context["event"] = " ".join(event)
        options = {"Births": " was born ", "Deaths": " died ", "Events": " "}

        query = (
            self.context["event"].strip(".") + options[self.section] + "on what date?"
        )

        reference = self.context["date"] + ", " + year.strip()

        super().__init__(
            name="date-based question answering",
            desc="get help answering a question",
            goal="to get the answer to the following question",
            query=query,
            reference=reference,
            topic=self.context["event"],
            subtopic="",
            tags="",
            static_reference=True,
            static_query=True,
        )
