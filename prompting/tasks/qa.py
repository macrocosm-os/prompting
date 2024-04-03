from prompting.tasks import Task


# Used to obtain the query (which is a question about the context)
QUERY_PROMPT_TEMPLATE = """\

You are a question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity. You will adhere to a word limit of 50 words. Ask a specific question about the following context:

# Context:
{context}
"""

# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\

You are a question-answering expert, focusing on delivering comprehensive and accurate questions with depth and clarity. Where applicable, you will include references to credible sources to support your answers. You will adhere to a word limit of 150 words.

Answer the following question in detail, utilizing the provided context.

# Context:
{context}

# Question:
{question}
"""


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

        self.query_prompt = QUERY_PROMPT_TEMPLATE.format(context=context.content)
        self.query = self.generate_query(llm_pipeline)

        self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(
            context=context.content, question=self.query
        )
        if create_reference:
            self.reference = self.generate_reference(llm_pipeline)

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
