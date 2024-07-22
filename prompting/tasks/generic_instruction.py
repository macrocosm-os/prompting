from prompting.llms.base_llm import BasePipeline
from prompting.shared.context import Context
from prompting.tasks import Task

QUERY_PROMPT_TEMPLATE = """\
You are a question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity. Your response contains only the question, nothing more, nothing less. You will adhere to a word limit of 100 words.
{context}
"""

REFERENCE_PROMPT_TEMPLATE = """\
Answer the following question.

# Question:
{query}"""


class GenericInstructionTask(Task):
    name = "generic"
    desc = "get help on answering a general instruction"
    goal = "to get the answer to the following instruction"
    challenge_type = "query"

    reward_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.25),
        dict(name="relevance", weight=0.75),
    ]
    penalty_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
    ]

    cleaning_pipeline = [
        dict(name="remove_quotes"),
        dict(name="prune_ending"),
        dict(name="remove_roles"),
    ]

    def __init__(
        self,
        llm_pipeline: BasePipeline,
        context: Context,
        create_reference: bool = True,
    ):
        self.context: Context = context

        self.query_prompt: str = QUERY_PROMPT_TEMPLATE.format(context=context.content)
        self.query: str = self.generate_query(llm_pipeline)

        self.reference_prompt: str = REFERENCE_PROMPT_TEMPLATE.format(query=self.query)
        if create_reference:
            self.reference = self.generate_reference(llm_pipeline)

        self.topic: str = context.title
        self.subtopic: str = context.topic
        self.tags: list[str] = context.tags
