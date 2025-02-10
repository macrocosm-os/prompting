import json
import re
import time

from loguru import logger

from shared.timer import Timer
from validator_api.chat_completion import chat_completion

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


async def make_api_call(messages, max_tokens, model=None, is_final_answer=False):
    logger.info(f"Making API call with messages: {messages}")
    response = None
    response_dict = None
    for attempt in range(3):
        try:
            response = await chat_completion(
                body={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "model": model,
                    "stream": False,
                    "task": "InferenceTask",
                    "sampling_parameters": {
                        "temperature": 0.5,
                        "max_new_tokens": 1000,
                        "top_p": 1,
                    },
                }
            )
            # return response.choices[0].message.content
            response_dict = parse_multiple_json(response.choices[0].message.content)[0]
            return response_dict
        except Exception as e:
            logger.error(f"Failed to get valid step back from miner: {e}")
            if attempt == 2:
                logger.exception(f"Error generating answer: {e}, RESPONSE DICT: {response_dict}")
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


async def generate_response(original_messages: list[dict[str, str]]):
    messages = [
        {
            "role": "system",
            "content": """You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

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
        }
    ]
    messages += original_messages
    messages += [
        {
            "role": "assistant",
            "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem.",
        }
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    for _ in range(MAX_THINKING_STEPS):
        with Timer() as timer:
            step_data = await make_api_call(messages, 300)
        thinking_time = timer.final_time
        total_thinking_time += thinking_time

        steps.append((f"Step {step_count}: {step_data['title']}", step_data["content"], thinking_time))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data["next_action"] == "final_answer" or not step_data.get("next_action"):
            break

        step_count += 1

        # Yield after each step
        yield steps, None

    # Generate final answer
    messages.append(
        {
            "role": "user",
            "content": "Please provide the final answer based on your reasoning above. You must return your answer in a valid json.",
        }
    )

    start_time = time.time()
    final_data = await make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    if final_data["title"] == "Error":
        steps.append(("Error", final_data["content"], thinking_time))
        raise ValueError("Failed to generate final answer: {final_data['content']}")

    steps.append(("Final Answer", final_data["content"], thinking_time))

    yield steps, total_thinking_time
