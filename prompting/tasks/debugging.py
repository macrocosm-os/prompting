import random
from typing import Optional
import bittensor as bt
from dataclasses import dataclass
from prompting.llms.base_llm import BasePipeline
from prompting.shared.context import Context
from prompting.tasks import Task
import difflib


def corrupt(
    code: str,
    n_remove: int = 0,
    n_swap: int = 0,
    seed: Optional[int] = None,
    sep: str = " ",
    min_length: int = 1,
    max_length: int = 10,
    remove_comment_lines: bool = False,
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

    def remove(
        code, n: int, sep: str = " ", min_length: int = 1, max_length: int = 10
    ) -> str:
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

        return sep.join([chunk for i, chunk in enumerate(chunks) if i not in indices])

    def swap(
        code: str, sep: str = " ", min_length: int = 1, max_length: int = 10
    ) -> str:
        """Swap two random chunks in the code. Chunks can be characters, words, or lines."""
        chunks: list[str] = code.split(sep) if sep else list(code)

        # select 2 random chunks to swap
        indices: list[int] = random.sample(
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
        code = remove(code, n=1, sep=sep, min_length=min_length, max_length=max_length)
    for i in range(n_swap):
        code = swap(code, sep=sep, min_length=min_length, max_length=max_length)

    return code


def diff(query: str, reference: str) -> str:
    """Get the diff between two strings."""
    return "\n".join(difflib.unified_diff(query.splitlines(), reference.splitlines()))


@dataclass
class DebuggingTask(Task):
    name = "debugging"
    desc = "get help with debugging"
    goal = "ask for help fixing broken code."

    reward_definition = [dict(name="diff", weight=1.0)]

    penalty_definition = []

    static_reference = True
    static_query = True

    def __init__(
        self,
        llm_pipeline: BasePipeline,
        context: Context,
        create_reference: bool = True,
    ):
        self.context: Context = context

        # No LLM involved in generating the query, we just apply some language-independent corruption to the code
        self.query: str = corrupt(
            context.content,
            n_remove=random.randint(1, 3),
            n_swap=random.randint(0, 2),
            sep=random.choices(["", " ", "\n"], weights=[0.3, 0.6, 0.1], k=1)[0],
        )
        self.reference: str = context.content
        self.delimiter: str = "```"
        self.topic: str = context.title
        self.subtopic: str = context.subtopic
        self.tags: list[str] = context.tags

    def format_challenge(self, challenge: str) -> str:
        return f"{challenge}\n{self.delimiter}\n{self.query}\n{self.delimiter}"
