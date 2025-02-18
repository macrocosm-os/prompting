import json
import random
import re
import time
from typing import ClassVar

from loguru import logger

from prompting.llms.apis.gpt_wrapper import LLMMessage, LLMMessages
from prompting.llms.apis.llm_wrapper import LLMWrapper
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from prompting.tasks.qa import WikiQuestionAnsweringTask
from shared.base import Context
from shared.timer import Timer

MAX_THINKING_STEPS = 10


def parse_multiple_json(api_response):
    """
    Parses a string containing multiple JSON objects and returns a list of dictionaries.

    Args:
        api_response (str): The string returned by the API containing JSON objects.

    Returns:
        list: A list of dictionaries parsed from the JSON objects.
    """
    # Regular expression pattern to match individual JSON objects
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)

    # Find all JSON object strings in the response
    json_strings = json_pattern.findall(api_response)

    parsed_objects = []
    for json_str in json_strings:
        try:
            # Replace escaped single quotes with actual single quotes
            json_str_clean = json_str.replace("\\'", "'")

            # Parse the JSON string into a dictionary
            obj = json.loads(json_str_clean)
            parsed_objects.append(obj)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON object: {e}")
            continue

    return parsed_objects


def make_api_call(messages, max_tokens, is_final_answer=False):
    # TOOD: Make this use local model to prevent relay mining
    for attempt in range(3):
        try:
            response = LLMWrapper.chat_complete(messages=LLMMessages(*messages))
            response_dict = parse_multiple_json(response)[0]
            return response_dict
        except Exception as e:
            if attempt == 2:
                logger.debug(f"ERROR GENERATING ANSWER. RESPONSE DICT: {response_dict}")
                if is_final_answer:
                    return {
                        "title": "Error",
                        "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}",
                    }
                else:
                    return {
                        "title": "Error",
                        "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                        "next_action": "final_answer",
                    }
            time.sleep(1)  # Wait for 1 second before retrying


def generate_response(prompt):
    messages = [
        LLMMessage(
            role="system",
            content="""You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

1. Provide a clear, concise title describing the current reasoning phase.
2. Elaborate on your thought process in the content section.
3. Decide whether to continue reasoning or provide a final answer.

Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Key Instructions:
- Employ at least 5 distinct reasoning steps.
- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
- Actively explore and evaluate alternative answers or approaches.
- Critically assess your own reasoning; identify potential flaws or biases.
- When re-examining, employ a fundamentally different approach or perspective.
- Utilize at least 3 diverse methods to derive or verify your answer.
- Incorporate relevant domain knowledge and best practices in your reasoning.
- Quantify certainty levels for each step and the final conclusion when applicable.
- Consider potential edge cases or exceptions to your reasoning.
- Provide clear justifications for eliminating alternative hypotheses.
- Output only one step at a time to ensure a detailed and coherent explanation.


Example of a valid JSON response:
```json
{
    "title": "Initial Problem Analysis",
    "content": "To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.",
    "next_action": "continue"
}```
""",
        )
    ]
    messages += [LLMMessage(role="user", content=prompt)]
    messages += [
        LLMMessage(
            role="assistant",
            content="Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem.",
        )
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    for _ in range(MAX_THINKING_STEPS):
        with Timer() as timer:
            step_data = make_api_call(messages, 300)
        thinking_time = timer.final_time
        total_thinking_time += thinking_time

        steps.append((f"Step {step_count}: {step_data['title']}", step_data["content"], thinking_time))

        messages.append(LLMMessage(role="assistant", content=json.dumps(step_data)))

        if step_data["next_action"] == "final_answer" or not step_data.get("next_action"):
            break

        step_count += 1

        # Yield after each step
        yield steps, None

    # Generate final answer
    messages.append(
        LLMMessage(
            role="user",
            content="Please provide the final answer based on your reasoning above. You must return your answer in a valid json.",
        )
    )

    start_time = time.time()
    final_data = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    if final_data["title"] == "Error":
        steps.append(("Error", final_data["content"], thinking_time))
        raise ValueError("Failed to generate final answer: {final_data['content']}")

    steps.append(("Final Answer", final_data["content"], thinking_time))

    yield steps, total_thinking_time


def execute_multi_step_reasoning(user_query):
    for steps, total_thinking_time in generate_response(user_query):
        if total_thinking_time is not None:
            logger.info(f"**Total thinking time: {total_thinking_time:.2f} seconds**")
    return steps, total_thinking_time


class MultiStepReasoningRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        RelevanceRewardModel(weight=1),
    ]


# Used to instruct the LLM to provide a good query when given a context
QUERY_SYSTEM_PROMPT = """\
You are a master of crafting intellectually stimulating questions that unfold across multiple sentences. Each question you generate should be structured as a brief narrative or scenario, where crucial information is deliberately distributed across multiple sentences. The complete question can only be understood and answered by carefully considering all the information provided across these sentences.

Your questions should:
1. Begin with context or background information
2. Introduce key variables or constraints in subsequent sentences
3. Present the actual question in the final sentence
4. Require analytical reasoning rather than mere fact recall
5. Draw from the provided context when available
6. Incorporate multiple related concepts or data points

EXAMPLE FORMATS:
✓ "The International Space Station orbits at an average height of 400km above Earth. At this height, it completes one orbit every 92 minutes. Assuming constant speed, how many kilometers does the ISS travel in one Earth day?"

✓ "A new streaming service launches with 500,000 subscribers in January. They observe that they lose 5% of their existing subscribers each month, but also gain 50,000 new subscribers in the same period. Their infrastructure costs increase by $100,000 for every 200,000 subscribers. What will their monthly infrastructure costs be after 6 months?"

✓ "The average American household generates 4.5 pounds of trash daily. Local recycling programs typically reduce landfill waste by 30%. Your city has just implemented a new composting initiative that diverts an additional 25% of waste from landfills. Considering there are 50,000 households in your city, how many pounds of waste would still reach landfills each week?"

AVOID:
- Single-sentence questions
- Questions answerable with simple facts
- Questions without context or background
- Obvious or straightforward calculations
- Questions that don't require analysis

Remember: The goal is to create questions where the context and parameters are revealed progressively, requiring the reader to integrate information across multiple sentences to fully understand and solve the problem. Make sure that the question is spread over at least 3 sentences.
"""

QUERY_PROMPT_TEMPLATE = """\
Ask a specific question about the following context:

#Context:
{context}

You must ask a question that can be answered by the context.
"""

SAMPLE_SYSTEM_PROMPTS = [
    """You are an LLM specialising in reasoning and solving complex questions. You will be given a chat interaction with a user and must answer appropriately.""",
    """You are a step-by-step problem solver. When given a complex question, you break it down into clear logical steps, showing your work and explaining your reasoning at each stage. You maintain a methodical approach to ensure accuracy.""",
    """You are an expert at mathematical and analytical reasoning. You excel at carefully parsing multi-part problems, identifying key information, and systematically working through solutions while clearly documenting your thought process.""",
]


class MultiStepReasoningTask(WikiQuestionAnsweringTask):
    """QuestionAnsweringTasks must be initialised with an LLM pipeline to generate query and reference plus
    context from a dataset to base the query on"""

    name: ClassVar[str] = "multi_step_reasoning"
    augmentation_system_prompt: ClassVar[str] = ""
    query: str | None = None
    reference: str | None = None

    def make_query(self, dataset_entry: Context):
        query_prompt = QUERY_PROMPT_TEMPLATE.format(context=dataset_entry.content)
        question = self.generate_query(messages=[QUERY_SYSTEM_PROMPT, query_prompt])
        msgs = [p + ". " if i < len(question.split(". ")) - 1 else p for i, p in enumerate(question.split(". ")) if p]
        self.messages = [{"role": "system", "content": random.choice(SAMPLE_SYSTEM_PROMPTS)}] + [
            {"role": random.choice("user", "assistant"), "content": msg} for msg in msgs
        ]
        return self.query

    def make_reference(self, dataset_entry: Context):
        steps, total_thinking_time = execute_multi_step_reasoning(user_query=self.query)
        self.reference = steps[-1][1]
        return self.reference
