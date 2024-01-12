import random
import bittensor as bt
from dataclasses import dataclass
from prompting.tasks import Task
import difflib

# The two options are:
# 1. Create a reference code and then introduce a bug to create the challenge code
# 2. Create a challenge code and then fix the bug to create the reference code

QUERY_SYSTEM_PROMPT = """\
You act as a coding teacher specializing in creating coding exercises by intentionally corrupting code snippets provided by the user. The purpose is to challenge the user to identify and fix the errors. When given a piece of code, analyze it, introduce errors or bugs, and then present the modified code as an exercise. The exercise will include the corrupted code and a question related to identifying or fixing the error. You should ensure that the errors introduced are logical or syntactical, suitable for educational purposes. It should not alter the core logic or purpose of the original code beyond recognition. Do not include comments or otherwise indicate in the code that it has been modified. The code should be within triple backticks (```).
"""

# Used to obtain the query (which is a question about the context)
QUERY_PROMPT_TEMPLATE = """\
Introduce a bug to the following {language} code snippet in triple backticks (```):

# Code:
{context}
"""


# Query is the student exercise containing only the broken code
# Reference is the solution to the exercise

# REFERENCE_PROMPT_TEMPLATE = """\
# You are an expert coding teacher, designed to help a student with a question.

# # Question:
# {query}
# """


def corrupt(
    code,
    n_remove=0,
    n_swap=0,
    seed=None,
    sep=" ",
    min_length=1,
    max_length=10,
    remove_comment_lines=False,
):
    """
    Corrupt a piece of code by removing and/or swapping chunks of it.
    TODO: Ignore comments and strings(?) when corrupting the code.

    Args:
        code (str): The code to corrupt.
        n_remove (int): The number of chunks to remove.
        n_swap (int): The number of chunks to swap.
        seed (int): The random seed to use.
        sep (str): The separator to use when splitting the code into chunks. Recommended values are '', ' ', '\n'.
        min_length (int): The minimum length of a chunk.
        max_length (int): The maximum length of a chunk.
    """

    # set seed for reproducibility
    random.seed(seed)

    assert n_remove + n_swap > 0, "Must specify at least one corruption type."

    def remove(code, n, sep=" ", min_length=1, max_length=10):
        """Remove n random chunks from the code. Chunks can be characters, words, or lines."""

        chunks = code.split(sep) if sep else list(code)

        # select n random chunks to remove
        indices = random.sample(
            [
                i
                for i, chunk in enumerate(chunks)
                if min_length <= len(chunk) <= max_length
            ],
            n,
        )
        bt.logging.info(
            f"Removing the following {len(indices)} chunks: {[chunks[i] for i in indices]} at indices {indices}"
        )

        return sep.join(
            [chunk for i, chunk in enumerate(chunks) if i not in indices]
        )

    def swap(code, sep=" ", min_length=1, max_length=10):
        """Swap two random chunks in the code. Chunks can be characters, words, or lines."""
        chunks = code.split(sep) if sep else list(code)

        # select 2 random chunks to swap
        indices = random.sample(
            [
                i
                for i, chunk in enumerate(chunks)
                if min_length <= len(chunk) <= max_length
            ],
            2,
        )

        bt.logging.info(
            f"Swapping chunk {chunks[indices[0]]!r} at index {indices[0]} with chunk {chunks[indices[1]]!r} at index {indices[1]}"
        )

        chunks[indices[0]], chunks[indices[1]] = (
            chunks[indices[1]],
            chunks[indices[0]],
        )

        return sep.join(chunks)

    # Do this at your peril. It doesn't catch multiline comments or strings.
    if remove_comment_lines:
        code = "\n".join(
            [
                line
                for line in code.splitlines()
                if not line.strip() or line.strip().startswith("#", "//")
            ]
        )

    # spread n corruptions across the code
    for i in range(n_remove):
        code = remove(
            code, n=1, sep=sep, min_length=min_length, max_length=max_length
        )
    for i in range(n_swap):
        code = swap(code, sep=sep, min_length=min_length, max_length=max_length)

    return code


def diff(query, reference):
    """Get the diff between two strings."""
    return "\n".join(
        difflib.unified_diff(query.splitlines(), reference.splitlines())
    )


@dataclass
class DebuggingTask(Task):
    reward_definition = [
        dict(name="diff", lines=False, threshold=0.5, weight=1.0),
        dict(name="relevance", threshold=None, weight=1.0),
    ]

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        # Query involves breaking the code somehow
        # self.query_system_prompt = QUERY_SYSTEM_PROMPT
        # self.query_prompt = QUERY_PROMPT_TEMPLATE.format(language=self.context['language'], context=self.context['code'])
        # query = self.generate_query(llm_pipeline)

        # No LLM involved in generating the query, we just apply some language-independent corruption to the code
        query = self.generate_query()
        reference = self.generate_reference()

        super().__init__(
            name="debugging",
            desc="get help with debugging",
            goal=f"ask for help fixing the broken piece of code. When asking for help do not adjust the code in any way.",
            query=query,
            reference=reference,
            topic=self.context["repo_name"],
            subtopic=self.context["path"],
            tags=self.context["language"],
        )

    def generate_query(
        self,
        llm=None,
        n_remove=1,
        n_swap=1,
        seed=0,
        sep="",
        min_length=1,
        max_length=10,
    ):
        self.query = corrupt(
            self.context["code"],
            n_remove=n_remove,
            n_swap=n_swap,
            seed=seed,
            sep=sep,
            min_length=min_length,
            max_length=max_length,
        )
        return self.query

    def generate_reference(self, llm=None):
        """Overrides the default reference generation to just return the reference code"""
        return self.context["code"]
