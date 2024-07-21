from dataclasses import dataclass
from prompting.tasks import Task


# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# TODO: Also add a query system prompt and a query prompt template
# TODO: Add the option to generate the summary query from the context. e.g. "the childhood of Abraham Lincoln" which is more specific than summarizing the entire article (Abraham Lincoln)

# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\
Summarize the following context in a concise and accurate manner:

## Context
{context}
"""


@dataclass
class SummarizationTask(Task):
    name = "summarization"
    desc = "get help with summarization"
    goal = "summarize the following topic"

    reward_definition = [
        dict(name="rouge", ngram="rouge-l", metric="f", weight=0.5),
        dict(name="relevance", weight=0.5),
    ]
    penalty_definition = [dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5)]

    # This is where you define cleaning procedures for the generation.
    # Can be used when wanting to clean the challenge.
    cleaning_pipeline = [
        dict(name="remove_quotes"),
        dict(name="prune_ending"),
        dict(name="remove_roles"),
    ]

    static_query = True

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        # Query is just the article title and section name
        self.query = context.title + ", " + context.topic

        self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(
            context=context.content
        )
        if create_reference:
            self.reference = self.generate_reference(llm_pipeline)

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags
