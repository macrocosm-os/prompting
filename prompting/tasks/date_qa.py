from dataclasses import dataclass
from prompting.tasks import Task
from prompting.cleaners.cleaner import CleanerPipeline

QUERY_SYSTEM_PROMPT = """You are a question creation expert. When asked to create a question, you use the context to make a specific question that would have the answer <date>. Your question should contain the topic."""
QUERY_PROMPT_TEMPLATE = """\
Create a question about {topic} that would have <date> as the answer using the following context:
topic: {topic}
context: {context}
"""
REFERENCE_PROMPT_TEMPLATE = """\
Your answer must include the following date: {date}.
Answer the following question using the provided context. 
Question: {query}
Context: {context}
"""

@dataclass
class DateQuestionAnsweringTask(Task):
    name = "date_qa"
    challenge_type = "query"
    clean_reference = False
    desc = "get help answering a specific date-based question"
    goal = "to get the answer to the following date-based question"
    reward_definition = [
        dict(name="date", weight=0.7),
        dict(name="rouge", weight=0.3),
    ]
    penalty_definition = []
    cleaning_pipeline = [
        #dict(name="remove_quotes"),
        #dict(name="remove_roles"),
        dict(name="remove_tags"), 
        dict(name="first_question"),
    ]
    static_reference = False

    def __init__(self, llm_pipeline, context, create_reference =True):
        self.context = context
        self.query_system_prompt = QUERY_SYSTEM_PROMPT
        self.query_prompt = QUERY_PROMPT_TEMPLATE.format(topic = context.title, context=context.content)
        self.query = self.generate_query(llm_pipeline)
        date = self.context.extra.get("date", None)
        self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(date = date, query = self.query, context = context.content)
        if create_reference:
            self.reference = self.generate_reference(llm_pipeline)
        self.topic = context.title
        self.subtopic = date
        self.tags = context.tags
