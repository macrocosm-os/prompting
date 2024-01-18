from dataclasses import dataclass
from prompting.tasks import Task
from transformers import Pipeline
from prompting.utils.clean_generation import GenerationCleaner


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
        dict(name="rouge", ngram="rouge-l", metric="f", weight=1.0),
        dict(name="relevance", threshold=None, weight=1.0),
    ]

    cleaner_pipeline = []

    def __init__(self, llm_pipeline: Pipeline, context: str, create_reference=True):
        NAME = "summarization"
        self.cleaner = GenerationCleaner()
        self.context = context

        self.query_prompt = None
        # NOTE: We do not perform an inference here and just use the article title as the query.
        # This is because the article title is usually a good summary of the article itself.
        # Query is just the article title.
        query = self.context["title"]

        self.reference_system_prompt = SUMMARIZATION_SYSTEM_PROMPT
        self.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(
            context=self.context["text"]
        )
        if create_reference:
            reference = self.generate_reference(llm=llm_pipeline)
            reference = self.cleaner.apply(generation=reference, task_name=NAME)

        else:
            reference = None

        super().__init__(
            name=NAME,
            desc="get help with summarization",
            goal="summarize the following topic",
            query=query,
            reference=reference,
            topic=self.context["title"],
            subtopic=self.context["categories"][0],
            tags=self.context["categories"],
        )
