import re
import bittensor as bt
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt
from prompting.tasks import Task
from typing import Tuple

CRITERIA_GENERATION_PROMPT = """\
We are brainstorming criteria with which to grade a language model on its responses in
diverse situations.
A ‘criteria‘ is some useful, real-world objective, and associated rubric for scores 1-5, that
tests a capability.

Please brainstorm a new criteria and scoring rubrics.
Be creative and create new but useful criteria that people in different settings or industries
might find practical.
Please format the output as same as the above examples with no extra or surrounding text.
Write [END] after you are done.
New Criteria:
"""


INSTRUCTION_GENERATION_PROMPT = """\
Your job is to generate a new novel problem and a response that is related to the given score
rubric.
The score rubric:
{CRITERIA}
* Problem
- The problem should inherently be related to the score criteria and score rubric given above.
Specifically, the score criteria should be the core attributes required to solve the problem.
- The problem itself should not be too generic or easy to solve.
- If the score rubric is related to logical abilities, generate problems that require math or
coding abilities.
- Try to make the person who might solve the problem not notice the existence of the score
rubric by not explicitly mentioning it, and also provide additional inputs and options if
needed.
- Assume a situation where a user is interacting with an AI model. The user would try to
ask in a first-person point of view, but not using terms like ”I”, ”A User” or ”You” in the
first sentence.
- Do not give a role to the AI, assume that the user is asking a question from his point of
view.
- Do not include any phrase related to AI model in the problem.
* Response
- The response should be a response that would get a score of 5 from the score rubric.
- The response should be as detailed as possible unless the score rubric is related to
conciseness or brevity. It should consist of multiple paragraphs, a list of items, or a
step-by-step reasoning process.
- The response should look like how a well-prompted GPT-4 would normally answer your
problem.
* Format
- DO NOT WRITE ANY GREETING MESSAGES, just write the problem and response
only.
- In front of the problem, append the phrase ”Problem:” and in front of the response, append
the phrase ”Response:”.
- Write in the order of ”Problem” - ”Response”, where the two items are separated by the
phrase ”[NEXT]”.
- Write [END] after you are done.
Data Generation:
"""


@dataclass
class GenericInstructionTask(Task):
    reward_definition = [
        dict(name="rouge", ngram="rouge-1", metric="f", weight=1.0),
        dict(name="relevance", threshold=None, weight=1.0),
    ]

    def __init__(self, llm_pipeline):
        super().__init__(
            name="generic_instruction",
            goal="to get the answer to a instruction",
            delimiter="```",
            reward_types=[
                "CRITERIA_REWARD",
            ],
            reward_threshold=0.5,
            use_challenge_as_prompt=True,
            desc="",
            topics={},
            topic="",
            subtopic="",
            challenge="",
            reference="",
            criteria="",
        )

        self.criteria = self.create_criteria(llm_pipeline)
        instruction, reference = self.create_instruction_and_reference(llm_pipeline)
        self.challenge = instruction
        self.reference = reference

    def extract_instruction_and_reference_from_text(self, text: str) -> Tuple[str, str]:
        # Split the text into problem and response using regular expression
        split_text = re.split(r"\nResponse:\n", text)

        # Extract problem and response
        problem = split_text[0].strip()
        response = split_text[1].strip()

        return problem, response

    def create_criteria(self, llm) -> str:
        bt.logging.debug("🎲 Creating a generic criteria-scoring rubric ...")

        # Generate a score rubric with defined criterias
        criteria_generation_response = llm(CRITERIA_GENERATION_PROMPT)
        return criteria_generation_response

    @retry(stop=stop_after_attempt(5))
    def create_instruction_and_reference(self, llm) -> Tuple[str, str]:
        try:
            bt.logging.debug("📋 🎯 Creating instruction and referece text...")

            if not self.criteria:
                raise ValueError(
                    "Criteria must be defined before creating a generic instruction."
                )

            # Create generic instruction with the score rubric
            instruction_generation_prompt_with_criteria = (
                INSTRUCTION_GENERATION_PROMPT.format(CRITERIA=self.criteria)
            )
            instruction_generation_response = llm(
                instruction_generation_prompt_with_criteria
            )

            # Extract generic instruction and reference response from the generated text
            (
                instruction,
                reference,
            ) = self.extract_instruction_and_reference_from_text(
                instruction_generation_response
            )

            return instruction, reference
        except Exception as e:
            bt.logging.error(
                f"Failed to create instruction and reference text: {e}, retrying..."
            )
            raise e
