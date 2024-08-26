from dataclasses import dataclass
import re
from typing import ClassVar

import numpy as np

from prompting.datasets.base import Context
from prompting.llms.base_llm import BasePipeline
from prompting.rewards.multi_choice import MultiChoiceRewardModel
from prompting.rewards.reward import BaseRewardConfig, WeightedRewardModel
from prompting.tasks.base_task import BaseTask
from prompting.utils.exceptions import TaskCreationError

# TODO: introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments

# Used to instruct the LLM to provide a good query when given a context.
QUERY_SYSTEM_PROMPT = """\
You are a multiple choice question-generating expert.
Provide 4 possible answers (multiple choice-style: A, B, C, D), one of which is correct.

Indicate the correct answer by placing an asterisk (*) at the beginning of the correct answer, the generated quiz must follow the same format:

A. [Incorrect Answer]
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
# QUERY_SYSTEM_PROMPT = """\
# You are a multiple choice question-generating expert, focusing on delivering comprehensive and accurate questions with depth and clarity for AI benchmarking purposes. The questions you generate should be specific and based on the context that is provided. Do not add any greetings or assistant messages before or after the question (e.g. Here's a question based on the context) as this does not bring any benefit to the question. The question should be self-contained. You will adhere to a word limit of 100 words for each question.

# In addition to the question, you will provide 4 possible answers (multiple choice-style), one of which is correct. Ensure that the correct answer is not too obvious, and that the incorrect answers are plausible. Be careful to maintain a neutral tone in the candidate answers, and randomly order the answers so that the correct answer is not always in the same position. Do not add any additional information after the answers.

# Indicate the correct answer by placing an asterisk (*) at the beginning of the correct answer, like this:

# A. [Correct Answer]
# *B. [Correct Answer]
# C. [Incorrect Answer]
# D. [Incorrect Answer]

# ## Example 1

# What is the capital of Texas?

# A. Paris
# B. London
# *C. Austin
# D. Houston

# ## Example 2

# Which of the following best describes the primary driving force behind protein folding?

# A. Covalent bond formation between amino acids
# *B. Hydrophobic interactions between nonpolar side chains
# C. Hydrogen bonds between the protein backbone and side chains
# D. Ionic interactions between charged side chains
# """

# Used to obtain the query (which is a question about the context)
# TODO: modulate difficulty "ask an {expert} question"
QUERY_PROMPT_TEMPLATE = """\
Create a multiple choice quiz based on the following context source from {source} about {title}:

#Context:
{context}
"""


class MultiChoiceRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=1.0, reward_model=MultiChoiceRewardModel()),
    ]


@dataclass
class MultiChoiceTask(BaseTask):
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""

    # specific pattern (semi-flexible) which detects multiple choices
    choices_pattern: ClassVar[str] = r"\n\s*(\*?\s*\W?[A-D]\W?)\s*(.*)"

    @classmethod
    def generate_query_reference(cls, llm_pipeline: BasePipeline, context: Context) -> tuple:
        query_prompt = QUERY_PROMPT_TEMPLATE.format(source=context.source, title=context.title, context=context.content)
        query_with_choices = cls.generate_query(llm_pipeline=llm_pipeline, messages=[query_prompt])
        query, reference = cls.extract_query_and_reference(query_with_choices)
        return query, reference

    @classmethod
    def extract_query_and_reference(cls, query_with_choices: str) -> tuple[str, str]:
        """Extract the query and reference answer from the generated query.

        To do this we extract the reference answer by searching for the choice with a * symbol,
        and then removing the * to form the query
        """
        # get the index of first occurrence of the choices
        index = re.search(cls.choices_pattern, query_with_choices).start()

        items, choices = list(
            zip(*re.findall(cls.choices_pattern, query_with_choices[index:]))
        )
        if len(choices) != 4:
            raise TaskCreationError(
                f"{cls.__name__} the number of choices is not 4 in query_with_choices:\n{query_with_choices}"
            )

        correct_item = [i for i, item in enumerate(items) if "*" in item]
        if len(correct_item) == 0:
            raise TaskCreationError(
                f"{cls.__name__} no reference was found in query_with_choices:\n{query_with_choices}"
            )
        elif len(correct_item) != 1:
            raise TaskCreationError(
                f"{cls.__name__} found multiple reference matches in query_with_choices:\n{query_with_choices}"
            )
        reference_label = choices[correct_item[0]]

        shuffled_choices, new_reference = cls.shuffle_choices(choices, reference_label)
        shuffled_query_with_choices = (
            query_with_choices[:index] + "\n\n" + shuffled_choices
        )
        return shuffled_query_with_choices, new_reference

    @classmethod
    def shuffle_choices(cls, choices: list[str], reference_label: str) -> tuple[str, str]:
        """Shuffle the choices and return the new reference.
        
        By itself, the LLM will almost always assign the reference to option B or C.
        This method overcomes the highly biased ordering by manually reshuffling the choices so that the reference is
        uniformly distributed across the options.

        Args:
            choices (list): list of choices
            reference_label (str): the reference answer label

        Returns:
            tuple[str, str]: The shuffled choices; The new reference, in {A, B, C, D}.
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
