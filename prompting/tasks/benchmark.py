import re
import bittensor as bt
from dataclasses import dataclass
from prompting.tasks import Task
from prompting.utils.exceptions import TaskCreationError

# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# Used to instruct the LLM to provide a good query when given a context
QUERY_SYSTEM_PROMPT = """\
You are a multiple choice question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity. The questions you generate should be specific and based on the context that is provided.
You will maintain a neutral tone in your questions.
You will adhere to a word limit of 100 words for each question.

In addition to the question, you will provide 4 possible answers (multiple choice-style), one of which is correct. Ensure that the correct answer is not too obvious, and that the incorrect answers are plausible. Be careful to maintain a neutral tone in the candidate answers, and randomly order the answers so that the correct answer is not always in the same position. Do not add any additional information after the answers.

Indicate the correct answer by placing an asterisk (*) at the beginning of the correct answer, like this:

A. [Correct Answer]
*B. [Correct Answer]
C. [Incorrect Answer]
D. [Incorrect Answer]
"""

# Used to obtain the query (which is a question about the context)
QUERY_PROMPT_TEMPLATE = """\
Create a multiple choice question based on the following context:

#Context:
{context}
"""


@dataclass
class BenchmarkingTask(Task):
    name = "benchmark"
    desc = "get help on answering a multiple choice question"
    goal = "to get the answer to the following multiple choice question"

    reward_definition = [
        dict(name="multiple_choice", ngram="multiple_choice", metric="f", weight=1.0),
    ]
    penalty_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=0.5),
    ]

    cleaning_pipeline = [
        dict(name="remove_quotes"),
        dict(name="prune_ending"),
        dict(name="remove_roles"),
        dict(name="remove_post_question_text"),
    ]

    static_reference = True
    challenge_type = "query"

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        self.query_system_prompt = QUERY_SYSTEM_PROMPT
        self.query_prompt = QUERY_PROMPT_TEMPLATE.format(context=context.content)

        query_with_choices = self.generate_query(llm_pipeline)

        self.query, self.reference = self.extract_query_and_reference(
            query_with_choices
        )

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags

    def extract_query_and_reference(self, query_with_choices):
        # Extract the query and reference answer from the generated query
        # we do this by
        # 1. stripping trailing newlines and whitespace
        # 2. regex matching the multiple choice options
        # 3. splitting the query from the reference answer

        # Match anything like this:
        # "*A. The capital of France is Paris."
        # "* B) The capital of France is Paris."
        # "* C The capital of France is Paris."
        # "*D The capital of France is Paris."
        # "* D. The capital of France is Paris."
        # "* (A) The capital of France is Paris."
        # "* B The capital of France is Paris."
        # "*C: The capital of France is Paris."
        # "*D The capital of France is Paris."
        pattern = r"\n\s*\*\s*\W?([A-D])\W?\s*(.*)"
        reference_matches = re.findall(pattern, query_with_choices)
        if len(reference_matches) == 0:
            raise TaskCreationError(
                f"{self.__class__.__name__} failed to extract query and reference from query_with_choices:\n{query_with_choices}"
            )
        elif len(reference_matches) > 1:
            raise TaskCreationError(
                f"{self.__class__.__name__} found multiple reference matches in query_with_choices:\n{query_with_choices}"
            )

        correct_letter, _ = reference_matches[0]
        # remove the asterisk which reveals the correct answer
        query = "\n".join(
            [line.strip("* ") for line in query_with_choices.splitlines()]
        )

        return query, correct_letter
