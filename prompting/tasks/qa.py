from dataclasses import dataclass
from prompting.tasks import Task

# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments
# TODO

# Used to instruct the LLM to provide a good query when given a context
QUERY_SYSTEM_PROMPT = """\
You are a question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity. The questions you generate should be based on the context that is provided.
You will maintain a neutral tone in your questions.
You will adhere to a word limit of 50 words for each question.
"""

# Used to obtain the query (which is a question about the context)
QUERY_PROMPT_TEMPLATE = """\
Ask a specific question about the following context:

#Context:
{context}
"""

# Used to instruct the LLM to provide a good answer to the query when given a context
REFERENCE_SYSTEM_PROMPT = """\
You are a question-answering expert, focusing on delivering comprehensive and accurate responses with depth and clarity.
You will maintain a neutral tone in your explanations.
You will adhere to a word limit of 150 words for each response. Where applicable, include references to credible sources to support your answers.
"""

# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\
Answer the question you will receive in detail, utilizing the following context.

#Context:
{context}

# Question:
{question}
"""


@dataclass
class QuestionAnsweringTask(Task):

    name = "question-answering"
    desc = "get help on answering a question"
    goal = "to get the answer to the following question"

    reward_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
        dict(name="relevance", weight=0.5),
    ]
    penalty_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
    ]

    cleaning_pipeline = [
        dict(name="remove_quotes"),
        dict(name="prune_ending"),
        dict(name="remove_roles"),
    ]

    def __init__(self, llm_pipeline, context, create_reference=True):


        self.context = context

        self.query_system_prompt = QUERY_SYSTEM_PROMPT
        self.query_prompt = QUERY_PROMPT_TEMPLATE.format(
            context = context.content
        )
        self.query = self.generate_query(llm_pipeline)

        self.reference_system_prompt = REFERENCE_SYSTEM_PROMPT
        self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(
            context = context.content, question = self.query
        )
        if create_reference:
            self.reference = self.generate_reference(llm_pipeline)

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
