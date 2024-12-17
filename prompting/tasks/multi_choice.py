import json
import random
from typing import ClassVar

import numpy as np

from shared.base import Context
from prompting.rewards.multi_choice import MultiChoiceRewardModel
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from prompting.tasks.base_task import BaseTextTask
from shared.exceptions import TaskCreationError

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
QUERY_SYSTEM_PROMPT = """Given the following input context, create a multiple-choice question based on the information provided. The question must have one correct answer and three incorrect answers.
Ensure the following:
1. The correct answer is derived from the input context.
2. All answer choices should have roughly the same character length. No answer should significantly stand out as longer or shorter.
3. The correct answer should not consistently be the longest option; it should only be the longest about 25% of the time.
4. Randomize answer length distribution across multiple samples.
5. The output format must match the example's output format.
[Example 1]
{
    "question": "Which of the following is not an element of the redistribution-with-growth policy approach?",
    "A": "minimum wage legislation",
    "B": "land reform",
    "C": "progressive taxation",
    "D": "increased access to education",
    "answer": "A"
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
[Example 3]
{
    "question": "What is the capital of Texas?",
    "A": "Paris",
    "B": "London",
    "C": "Austin",
    "D": "Houston",
    "answer": "C"
}
[Example 4]
{
    "question": "What interior discipline must be adopted to achieve spiritual liberation within Sikhism?",
    "A": "Remembering the Divine Name",
    "B": "Meditating on the sacred hymns",
    "C": "Remembering that death is inevitable",
    "D": "Meditating on the goodness of the created world",
    "answer": "A"
}"""

# Used to obtain the query (which is a question about the context).
# TODO: modulate difficulty "ask an {expert} question".
QUERY_PROMPT_TEMPLATE = """\
Create a multiple choice quiz based on the following context source from {source} about {title}:

[Input Context]
{context}
"""


class MultiChoiceRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        MultiChoiceRewardModel(weight=1.0),
    ]


class MultiChoiceTask(BaseTextTask):
    name: ClassVar[str] = "multi_choice"
    query_system_prompt: ClassVar[str] = QUERY_SYSTEM_PROMPT
    augmentation_system_prompt: ClassVar[str] = ""
    llm_model_id: str | None = None

    # Specific pattern (semi-flexible) which detects multiple choices.
    choices_pattern: ClassVar[str] = r"\n\s*(\*?\s*\W?[A-D]\W?)\s*(.*)"

    def make_query(self, dataset_entry: Context) -> tuple[str, str]:
        query_prompt = QUERY_PROMPT_TEMPLATE.format(
            source=dataset_entry.source, title=dataset_entry.title, context=dataset_entry.content
        )
        query_with_choices = self.generate_query(messages=[query_prompt])
        self.query, self.reference = self.extract_query_and_reference(query_with_choices)
        self.query = self.post_process_qa(self.query)
        return self.query

    def post_process_qa(self, query: str) -> str:
        options = query.split("?")[2].split("\n")
        cleaned_options = [item.strip() for item in options if item.strip() and item.strip() != "Answer:"]
        letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
        try:
            int(cleaned_options[letter_to_index.get(self.reference)].split(". ")[1])
        except Exception:
            return query
        new_idx = random.randint(0, 3)
        answer = int(cleaned_options[letter_to_index.get(self.reference)].split(". ")[1])
        step = random.randint(1, 10)
        new_options = [int(answer) + (i - new_idx) * step for i in range(4)]
        new_options = [opt for opt in new_options if opt != answer]
        letter_options = ["A. ", "B. ", "C. ", "D. "]
        available_letters = [opt for opt in letter_options if f"{self.reference}. " not in opt]
        random.shuffle(available_letters)
        random.shuffle(new_options)
        new_options = [available_letters[i] + str(new_options[i]) for i in range(3)]
        new_options.append(self.reference + ". " + str(answer))
        new_options = sorted(new_options, key=lambda x: x.split(". ")[0])
        new_options.append("Answer:")
        options_string = "\n".join(new_options)
        new_query = "?".join(query.split("?")[:2]) + "?\n" + options_string
        return new_query

    def make_reference(self, dataset_entry: Context) -> str:
        return self.reference

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
