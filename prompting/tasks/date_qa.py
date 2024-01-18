from dataclasses import dataclass
from prompting.tasks import Task
from prompting.utils.clean_generation import GenerationCleaner


@dataclass
class DateQuestionAnsweringTask(Task):
    reward_definition = [
        dict(name="date", weight=1),
    ]

    def __init__(self, llm_pipeline, context, create_reference=True):
        NAME = "date-based question answering"
        self.cleaner = GenerationCleaner()
        self.context = context
        self.section = self.context["section"]
        year, _, *event = self.context["event"].split()
        self.context["event"] = " ".join(event)
        options = {"Births": " was born ", "Deaths": " died ", "Events": " "}

        query = (
            self.context["event"].strip(".") + options[self.section] + "on what date?"
        )
        # query = self.cleaner.apply(generation=query, task_name = NAME) #Might not want to apply cleaning to query.

        reference = self.context["date"] + ", " + year.strip()
        reference = self.cleaner.apply(generation=reference, task_name=NAME)

        super().__init__(
            name=NAME,
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
