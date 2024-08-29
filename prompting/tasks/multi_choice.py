import json
from typing import ClassVar

import numpy as np

from prompting.datasets.base import Context
from prompting.rewards.multi_choice import MultiChoiceRewardModel
from prompting.rewards.reward import BaseRewardConfig, WeightedRewardModel
from prompting.tasks.base_task import BaseTask
from prompting.utils.exceptions import TaskCreationError

# TODO: Introduce criteria for the query and reference answer (length, layout, etc.) and make these arguments.

MINER_EXAMPLE_1_SHOT = """\
[Example 1]
What is the capital of Texas?
A. Paris
B. London
C. Austin
D. Houston
Answer: C
"""

# Used to instruct the LLM to provide a query when given a context.
QUERY_SYSTEM_PROMPT = """\
You are a multiple choice quiz-generating expert.
Based on the input context, you must generate the question, exactly 4 possible answers (A, B, C, D), and the correct answer letter.

[Example 1]
{
    "question": "What is the capital of Texas?",
    "A": "Paris",
    "B": "London",
    "C": "Austin",
    "D": "Houston",
    "answer": "C"
}

[Example 2]
{
    "question": "Which of the following best describes the primary driving force behind protein folding?",
    "A": "Covalent bond formation between amino acids",
    "B": "Hydrophobic interactions between nonpolar side chains",
    "C": "Hydrogen bonds between the protein backbone and side chains",
    "D": "Ionic interactions between charged side chains",
    "answer": "B"
}
"""

# Used to obtain the query (which is a question about the context).
# TODO: modulate difficulty "ask an {expert} question".
QUERY_PROMPT_TEMPLATE = """\
Create a multiple choice quiz based on the following context source from {source} about {title}:

[Input Context]
{context}
"""


class MultiChoiceRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[WeightedRewardModel]] = [
        WeightedRewardModel(weight=1.0, reward_model=MultiChoiceRewardModel()),
    ]


class MultiChoiceTask(BaseTask):
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""

    # Specific pattern (semi-flexible) which detects multiple choices.
    choices_pattern: ClassVar[str] = r"\n\s*(\*?\s*\W?[A-D]\W?)\s*(.*)"

    def make_query(self, context: Context) -> tuple[str, str]:
        query_prompt = QUERY_PROMPT_TEMPLATE.format(source=context.source, title=context.title, context=context.content)
        query_with_choices = self.generate_query(messages=query_prompt)
        self.query, self.reference = self.extract_query_and_reference(query_with_choices)
        return self.query

    def make_reference(self, context: Context) -> str:
        return self.reference

    @classmethod
    def extract_query_and_reference(self, query_with_choices: str) -> tuple[str, str]:
        """
        Detects JSON within a string, parses it into a dictionary,
        and validates that the dictionary contains the required fields:
        "question", "answer", "A", "B", "C", and "D".

        Args:
            json_string (str): The string containing the JSON data, possibly with extra text.

        Returns:
            dict: The parsed and validated dictionary.

        Raises:
            ValueError: If JSON extraction or parsing fails, or required fields are missing.
        """

        # Regular expression pattern to match JSON object in the string.
        def extract_json_from_string(string: str):
            start = string.find("{")
            end = string.rfind("}") + 1
            if start != -1 and end != -1:
                json_string = string[start:end]
                try:
                    return json.loads(json_string)
                except json.JSONDecodeError:
                    pass
            return None

        quiz_data = extract_json_from_string(query_with_choices)
        if not quiz_data:
            raise TaskCreationError(f"No JSON object could be found in the provided string: {query_with_choices}.")

        required_fields = ["question", "answer", "A", "B", "C", "D"]

        # Check for missing fields.
        for field in required_fields:
            if field not in quiz_data:
                raise TaskCreationError(f"Missing required field: '{field}'")

        # Answer must be exactly one of the choices.
        if quiz_data["answer"] not in ("A", "B", "C", "D"):
            raise TaskCreationError(f"Invalid answer: '{quiz_data['answer']}'")

        quiz, reference = self.shuffle_and_format(quiz_data)
        return quiz, reference

    def shuffle_and_format(self, quiz_data: dict[str, str]) -> tuple[str, str]:
        """Shuffles the choices and formats them into a string with the question.

        Args:
            quiz_data (dict): The dictionary containing the quiz data.

        Returns:
            str: The formatted string with the question and shuffled choices.
        """
        # Extract choices and the correct answer.
        choices = ["A", "B", "C", "D"]
        choice_texts = [quiz_data[choice] for choice in choices]
        correct_answer = quiz_data["answer"]

        # Shuffle the choices.
        shuffled_choices = list(zip(choices, np.random.permutation(choice_texts)))

        # Determine the new correct answer after shuffling.
        new_reference = [choice for choice, text in shuffled_choices if text == quiz_data[correct_answer]][0]

        # Format the shuffled question and choices.
        prompt: list[str] = []
        prompt.append(f"{MINER_EXAMPLE_1_SHOT}\n")
        prompt.append(f"[Input Question]\n{quiz_data['question']}\n\n")
        prompt.append("\n".join([f"{choice}. {text}" for choice, text in shuffled_choices]))
        prompt.append("\nAnswer: ")

        return "".join(prompt), new_reference
