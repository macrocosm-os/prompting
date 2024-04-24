from dataclasses import dataclass
from prompting.tasks import Task
from prompting.cleaners.cleaner import CleanerPipeline


SECTION_MESSAGES = {"Births": " was born ", "Deaths": " died ", "Events": " "}


@dataclass
class DateQuestionAnsweringTask(Task):
    name = "date_qa"
    desc = "get help answering a specific date-based question"
    goal = "to get the answer to the following date-based question"
    reward_definition = [
        dict(name="date", weight=1.0),
    ]
    penalty_definition = []
    cleaning_pipeline = [
        dict(name="remove_quotes"),
        dict(name="remove_roles"),
    ]
    static_reference = True
    static_query = True

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        self.query = (
            context.content + SECTION_MESSAGES[context.topic] + "on what exact date?"
        )
        self.reference = self.context.title.replace("_", " ") + ", " + context.subtopic

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
