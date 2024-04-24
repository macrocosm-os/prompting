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
    challenge_type = "query"
    name = "generic"
    desc = "get help on answering a general instruction"
    goal = "to get the answer to the following instruction"

    reward_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.75),
        dict(name="relevance", weight=0.25),
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

        self.query_prompt = QUERY_PROMPT_TEMPLATE.format(context=context.content)
        self.query = self.generate_query(llm_pipeline)

        self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(query=self.query)
        if create_reference:
            self.reference = self.generate_reference(llm_pipeline)

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
