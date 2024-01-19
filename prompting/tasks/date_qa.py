from dataclasses import dataclass
from prompting.tasks import Task
from prompting.cleaners.cleaner import CleanerPipeline


@dataclass
class DateQuestionAnsweringTask(Task):
    reward_definition = [
        dict(name="date", weight=1),
    ]
    penalty_definition = []

    def __init__(self, llm_pipeline, context, create_reference=True):
        
        
        self.name = "date-based question answering"
        self.desc = "get help answering a specific date-based question"
        self.goal = "to get the answer to the following date-based question"
        
        self.cleaning_pipeline = [
            dict(name="remove_quotes"),
            dict(name="remove_roles"),
        ]
        self.context = context
        
        # The section is in {"Births", "Deaths", "Events"}
        section = self.context["section"]
        year, _, *event = self.context["event"].split()
        self.context["event"] = " ".join(event)
        
        options = {'Births':' was born ', 'Deaths':' died ', 'Events':' '}
        
        self.query = self.context["event"].strip(".") + options[section] + 'on what exact date?'
        self.reference = self.context["date"] + ", " + year.strip()

        self.topic = section
        self.subtopic = year
        self.tags = []
        self.static_reference = True
        self.static_query = True

