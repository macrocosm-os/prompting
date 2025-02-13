import asyncio
import json
import random
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

    if len(parsed_objects) == 0:
        logger.error(
            f"No valid JSON objects found in the response - couldn't parse json. The miner response was: {api_response}"
        )
        return None
    if (
        not parsed_objects[0].get("title")
        or not parsed_objects[0].get("content")
        or not parsed_objects[0].get("next_action")
    ):
        logger.error(
            f"Invalid JSON object found in the response - field missing. The miner response was: {api_response}"
        )
        return None
    return parsed_objects


async def make_api_call(messages, max_tokens, model=None, is_final_answer=False):
    ATTEMPTS_PER_STEP = 10
    logger.info(f"Making API call with messages: {messages}")

    async def single_attempt():
        try:
            response = await chat_completion(
                body={
                    "messages": messages,
                    "model": model,
                    "task": "InferenceTask",
                    "test_time_inference": True,
                    "mixture": False,
                    "sampling_parameters": {
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "do_sample": True,
                        "max_new_tokens": 2048,
                    },
                    "seed": (seed := random.randint(0, 1000000)),
                }
            )
            logger.debug(
                f"Making API call with\n\nMESSAGES: {messages}\n\nSEED: {seed}\n\nRESPONSE: {response.choices[0].message.content}"
            )
            response_dict = parse_multiple_json(response.choices[0].message.content)[0]
            return response_dict
        except Exception as e:
            logger.error(f"Failed to get valid response: {e}")
            return None

    # Create three concurrent tasks
    try:
        return await single_attempt()
    except Exception as e:
        if is_final_answer:
            return {
                "title": "Error",
                "content": f"Failed to generate final answer. Error: {e}",
            }
        else:
            return {
                "title": "Error",
                "content": f"Failed to generate step. Error: {e}",
                "next_action": "final_answer",
            }


async def generate_response(original_messages: list[dict[str, str]], model: str = None):
    messages = [
        {
            "role": "system",
            "content": """You are a world-class expert in analytical reasoning and problem-solving. Your task is to break down complex problems through rigorous step-by-step analysis, carefully examining each aspect before moving forward. For each reasoning step:

OUTPUT FORMAT:
Return a JSON object with these required fields:
{
    "title": "Brief, descriptive title of current reasoning phase",
    "content": "Detailed explanation of your analysis",
    "next_action": "continue" or "final_answer"
}

REASONING PROCESS:
1. Initial Analysis
   - Break down the problem into core components
   - Identify key constraints and requirements
   - List relevant domain knowledge and principles

2. Multiple Perspectives
   - Examine the problem from at least 3 different angles
   - Consider both conventional and unconventional approaches
   - Identify potential biases in initial assumptions

3. Exploration & Validation
   - Test preliminary conclusions against edge cases
   - Apply domain-specific best practices
   - Quantify confidence levels when possible (e.g., 90% certain)
   - Document key uncertainties or limitations

4. Critical Review
   - Actively seek counterarguments to your reasoning
   - Identify potential failure modes
   - Consider alternative interpretations of the data/requirements
   - Validate assumptions against provided context

5. Synthesis & Refinement
   - Combine insights from multiple approaches
   - Strengthen weak points in the reasoning chain
   - Address identified edge cases and limitations
   - Build towards a comprehensive solution

REQUIREMENTS:
- Each step must focus on ONE specific aspect of reasoning
- Explicitly state confidence levels and uncertainty
- When evaluating options, use concrete criteria
- Include specific examples or scenarios when relevant
- Acknowledge limitations in your knowledge or capabilities
- Maintain logical consistency across steps
- Build on previous steps while avoiding redundancy

CRITICAL THINKING CHECKLIST:
✓ Have I considered non-obvious interpretations?
✓ Are my assumptions clearly stated and justified?
✓ Have I identified potential failure modes?
✓ Is my confidence level appropriate given the evidence?
✓ Have I adequately addressed counterarguments?

Remember: Quality of reasoning is more important than speed. Take the necessary steps to build a solid analytical foundation before moving to conclusions.""",
        }
    ]
    messages += original_messages
    messages += [
        {
            "role": "assistant",
            "content": "I understand. I will now analyze the problem systematically, following the structured reasoning process while maintaining high standards of analytical rigor and self-criticism.",
        }
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    for _ in range(MAX_THINKING_STEPS):
        with Timer() as timer:
            step_data = await make_api_call(messages, 300, model=model)
        thinking_time = timer.final_time
        total_thinking_time += thinking_time

        steps.append((f"Step {step_count}: {step_data['title']}", step_data["content"], thinking_time))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data["next_action"] == "final_answer" or not step_data.get("next_action"):
            break

        step_count += 1
        yield steps, None

    messages.append(
        {
            "role": "user",
            "content": """Based on your thorough analysis, please provide your final answer. Your response should:
1. Clearly state your conclusion
2. Summarize the key supporting evidence
3. Acknowledge any remaining uncertainties
4. Include relevant caveats or limitations

Return your answer in the same JSON format as previous steps.""",
        }
    )

    start_time = time.time()
    final_data = await make_api_call(messages, 200, is_final_answer=True, model=model)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    if final_data["title"] == "Error":
        steps.append(("Error", final_data["content"], thinking_time))
        raise ValueError(f"Failed to generate final answer: {final_data['content']}")

    steps.append(("Final Answer", final_data["content"], thinking_time))

    yield steps, total_thinking_time
