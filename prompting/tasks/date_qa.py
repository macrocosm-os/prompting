import bittensor as bt
from dataclasses import dataclass
from prompting.tasks import Task
import textwrap


@dataclass
class DateQuestionAnsweringTask(Task):
    reward_definition = [
        dict(name="rouge", ngram="rouge-l", metric="f", weight=1.0),
    ]
    # # Used to obtain the question [we actually can use a subset of the context]
    # query_template = """\
    # Ask a specific question about the date of the following event:

    # #Event:
    # {context}
    # """

    # # # Used to obtain reference answer
    # # reference_prompt_template = """\
    # # Answer the question you will receive in detail, utilizing the following context.

    # # #Context:
    # # {context}

    # # # Question:
    # # {question}
    # # """

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context
        year, _, *event = self.context["event"].split()
        self.context["event"] = " ".join(event)
        query = self.context["event"].strip(".") + " on what date?"
        reference = self.context["date"] + ", " + year.strip()
        super().__init__(
            name="date_qa",
            desc="get help on answering a question",
            goal=f"to get the answer to the following question",
            query=query,
            reference=reference,
            topic=self.context["event"],
            subtopic="",
            tags="",
            static_reference=True,
            static_query=True,
        )
