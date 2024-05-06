from dataclasses import dataclass
from prompting.tasks import Task
from prompting.cleaners.cleaner import CleanerPipeline

QUERY_SYSTEM_PROMPT = """You are a question creation expert. When asked to create a question, you use the context to make a specific question that would have the answer <date>."""
QUERY_PROMPT_TEMPLATE = """\
Create a question that would have <date> as the answer using the following context:
{context}
"""

@dataclass
class DateQuestionAnsweringTask(Task):
    name = "date_qa"
    challenge_type = 'query'
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

    def __init__(self, llm_pipeline, context, create_reference =True):
        self.context = context
        self.query_system_prompt = QUERY_SYSTEM_PROMPT
        self.query_prompt = QUERY_PROMPT_TEMPLATE.format(context=context.content[1])
        self.query = self.generate_query(llm_pipeline)
        self.reference = self.context['content'][0]

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
