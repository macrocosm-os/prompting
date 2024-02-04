from dataclasses import dataclass
from prompting.tasks import Task
from transformers import Pipeline


# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# TODO: Also add a query system prompt and a query prompt template
# TODO: Add the option to generate the summary query from the context. e.g. "the childhood of Abraham Lincoln" which is more specific than summarizing the entire article (Abraham Lincoln)

# Used to instruct the LLM to provide a good answer to the query when given a context
SUMMARIZATION_SYSTEM_PROMPT = """\
You are a summarization AI assistant. You make excellent and concise summaries that adhere to your given instructions.
You will maintain a neutral tone in your summaries.
You will adhere to a word limit of 250 words for each response.
"""

# Used to obtain reference answer
REFERENCE_PROMPT_TEMPLATE = """\
Summarize the following context in a concise and accurate manner:

## Context
{context}
"""


@dataclass
class SummarizationTask(Task):
    
    reward_definition = [
        dict(name="rouge", ngram="rouge-l", metric="f", weight=0.5),
        dict(name="relevance", threshold=None, weight=0.5),
    ]
    penalty_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=1.0)
    ]

    def __init__(self, llm_pipeline: Pipeline, context: str, create_reference=True):

        self.name = "summarization"
        self.desc = "get help with summarization"
        self.goal = "summarize the following topic"

        self.context = context


        # This is where you define cleaning procedures for the generation.
        # Can be used when wanting to clean the challenge.
        self.cleaning_pipeline = [
            dict(name="remove_quotes"),
            dict(name="prune_ending"),
            dict(name="remove_roles"),
        ]

        self.context = context

        self.query_prompt = None
        # NOTE: We do not perform an inference here and just use the article title as the query.
        # This is because the article title is usually a good summary of the article itself.
        # Query is just the article title.
        self.query = self.context["title"] + ', ' + self.context.topic

        self.reference_system_prompt = SUMMARIZATION_SYSTEM_PROMPT
        self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(
            context = self.context["text"]
        )
        if create_reference:
            self.reference = self.generate_reference(llm_pipeline)

        self.topic = self.context["title"]
        self.subtopic = self.context["categories"][0]
        self.tags = self.context["categories"]
        self.static_query = True

