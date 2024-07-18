import re
import numpy as np
import bittensor as bt
from dataclasses import dataclass
from prompting.tasks import Task
from prompting.utils.exceptions import TaskCreationError

# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# Used to instruct the LLM to provide a good query when given a context
QUERY_SYSTEM_PROMPT = """\
You are a multiple choice question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity for AI benchmarking purposes. The questions you generate should be specific and based on the context that is provided. Do not add any greetings or assistant messages before or after the question (e.g. Here's a question based on the context) as this does not bring any benefit to the question. The question should be self-contained. You will adhere to a word limit of 100 words for each question.

In addition to the question, you will provide 4 possible answers (multiple choice-style), one of which is correct. Ensure that the correct answer is not too obvious, and that the incorrect answers are plausible. Be careful to maintain a neutral tone in the candidate answers, and randomly order the answers so that the correct answer is not always in the same position. Do not add any additional information after the answers.

Indicate the correct answer by placing an asterisk (*) at the beginning of the correct answer, like this:

A. [Correct Answer]
*B. [Correct Answer]
C. [Incorrect Answer]
D. [Incorrect Answer]

## Example 1

What is the capital of Texas?

A. Paris
B. London
*C. Austin
D. Houston

## Example 2

Which of the following best describes the primary driving force behind protein folding?

A. Covalent bond formation between amino acids
*B. Hydrophobic interactions between nonpolar side chains
C. Hydrogen bonds between the protein backbone and side chains
D. Ionic interactions between charged side chains
"""

# Used to obtain the query (which is a question about the context)
# TODO: modulate difficulty "ask an {expert} question"
QUERY_PROMPT_TEMPLATE = """\
Create a multiple choice question based on the following context source from {source} about {title}:

#Context:
{context}
"""


@dataclass
class BenchmarkingTask(Task):
    name = "benchmark"
    desc = "get help on answering a multiple choice question"
    goal = "to get the answer to the following multiple choice question"

    reward_definition = [
        dict(name="multiple_choice", weight=1.0),
    ]
    penalty_definition = []
    cleaning_pipeline = []

    static_reference = True
    challenge_type = "query"

    # specific pattern (semi-flexible) which detects multiple choices
    choices_pattern = r"\n\s*(\*?\s*\W?[A-D]\W?)\s*(.*)"

    def __init__(self, llm_pipeline, context, create_reference=True):
        self.context = context

        self.query_system_prompt = QUERY_SYSTEM_PROMPT
        self.query_prompt = QUERY_PROMPT_TEMPLATE.format(
            context=context.content, title=context.title, source=context.source
        )

        query_with_choices = self.generate_query(llm_pipeline)

        self.query, self.reference = self.extract_query_and_reference(
            query_with_choices
        )

        self.topic = context.title
        self.subtopic = context.topic
        self.tags = context.tags

    def extract_query_and_reference(self, query_with_choices: str) -> Tuple[str, str]:
        """
        Extract the query and reference answer from the generated query.
        To do this we extract the reference answer by searching for the choice with a * symbol,
        and then removing the * to form the query
        """
        # get the index of first occurrence of the choices
        index = re.search(self.choices_pattern, query_with_choices).start()

        items, choices = list(
            zip(*re.findall(self.choices_pattern, query_with_choices[index:]))
        )
        if len(choices) != 4:
            raise TaskCreationError(
                f"{self.__class__.__name__} the number of choices is not 4 in query_with_choices:\n{query_with_choices}"
            )

        correct_item = [i for i, item in enumerate(items) if "*" in item]
        if len(correct_item) == 0:
            raise TaskCreationError(
                f"{self.__class__.__name__} no reference was found in query_with_choices:\n{query_with_choices}"
            )
        elif len(correct_item) != 1:
            raise TaskCreationError(
                f"{self.__class__.__name__} found multiple reference matches in query_with_choices:\n{query_with_choices}"
            )
        reference_label = choices[correct_item[0]]

        shuffled_choices, new_reference = self.shuffle_choices(choices, reference_label)
        shuffled_query_with_choices = (
            query_with_choices[:index] + "\n\n" + shuffled_choices
        )
        return shuffled_query_with_choices, new_reference

    def shuffle_choices(
        self, choices: List[str], reference_label: str
    ) -> Tuple[str, str]:
        """Shuffle the choices and return the new reference. By itself, the LLM will almost always assign the reference to option B or C.
        This method overcomes the highly biased ordering by manually reshuffling the choices so that the reference is uniformly distributed across the options.

        Args:
            choices (list): list of choices
            reference_label (str): the reference answer label

        Returns:
            str: the shuffled choices
            str: the new reference, in {A, B, C, D}
        """
        # shuffle the choices
        shuffled_choice_list = list(
            zip("ABCD", np.random.choice(choices, len(choices), replace=False))
        )

        # match the reference and get the letter
        new_reference = [c[0] for c in shuffled_choice_list if c[1] == reference_label][
            0
        ]

        # reconstruct the shuffled choices (without indicating which is correct)
        shuffled_choices = "\n".join([f"{c[0]}. {c[1]}" for c in shuffled_choice_list])

        return shuffled_choices, new_reference
