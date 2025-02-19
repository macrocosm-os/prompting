# import asyncio
# import json
# import random
# import re
# import time

# from loguru import logger

# from prompting.llms.apis.llm_messages import LLMMessage, LLMMessages
# from prompting.llms.apis.llm_wrapper import LLMWrapper
# from shared.timer import Timer
# from validator_api.chat_completion import chat_completion

# MAX_THINKING_STEPS = 10
# ATTEMPTS_PER_STEP = 10


# def parse_multiple_json(api_response):
#     """
#     Parses a string containing multiple JSON objects and returns a list of dictionaries.

#     Args:
#         api_response (str): The string returned by the API containing JSON objects.

#     Returns:
#         list: A list of dictionaries parsed from the JSON objects.
#     """
#     # Regular expression pattern to match individual JSON objects
#     json_pattern = re.compile(r"\{.*?\}", re.DOTALL)

#     # Find all JSON object strings in the response
#     json_strings = json_pattern.findall(api_response)

#     parsed_objects = []
#     for json_str in json_strings:
#         try:
#             # Replace escaped single quotes with actual single quotes
#             json_str_clean = json_str.replace("\\'", "'")

#             # Parse the JSON string into a dictionary
#             obj = json.loads(json_str_clean)
#             parsed_objects.append(obj)
#         except json.JSONDecodeError as e:
#             print(f"Failed to parse JSON object: {e}")
#             continue

#     if len(parsed_objects) == 0:
#         logger.error(
#             f"No valid JSON objects found in the response - couldn't parse json. The miner response was: {api_response}"
#         )
#         return None
#     if (
#         not parsed_objects[0].get("title")
#         or not parsed_objects[0].get("content")
#         or not parsed_objects[0].get("next_action")
#     ):
#         logger.error(
#             f"Invalid JSON object found in the response - field missing. The miner response was: {api_response}"
#         )
#         return None
#     return parsed_objects


# async def make_api_call(
#     messages, model=None, is_final_answer: bool = False, use_miners: bool = True, target_uids: list[str] = None
# ):
#     async def single_attempt():
#         try:
#             if use_miners:
#                 response = await chat_completion(
#                     body={
#                         "messages": messages,
#                         "model": model,
#                         "task": "InferenceTask",
#                         "test_time_inference": True,
#                         "mixture": False,
#                         "sampling_parameters": {
#                             "temperature": 0.5,
#                             "max_new_tokens": 1500,  # Increased token limit for longer response
#                         },
#                         "seed": random.randint(0, 1000000),
#                     },
#                     num_miners=3,
#                     uids=target_uids,
#                 )
#                 response_str = response.choices[0].message.content
#             else:
#                 logger.debug("Using SN19 API for inference in MSR")
#                 response_str = LLMWrapper.chat_complete(
#                     messages=LLMMessages(*[LLMMessage(role=m["role"], content=m["content"]) for m in messages]),
#                     model=model,
#                     temperature=0.5,
#                     max_tokens=2000,
#                 )

#             logger.debug(f"Making API call with\n\nMESSAGES: {messages}\n\nRESPONSE: {response_str}")
#             return response_str
#         except Exception as e:
#             logger.warning(f"Failed to get valid response: {e}")
#             return None

#     # When not using miners, let's try and save tokens
#     if not use_miners:
#         for _ in range(ATTEMPTS_PER_STEP):
#             try:
#                 result = await single_attempt()
#                 if result is not None:
#                     return result
#                 else:
#                     logger.error("Failed to get valid response")
#                     continue
#             except Exception as e:
#                 logger.error(f"Failed to get valid response: {e}")
#                 continue
#     else:
#         # when using miners, we try and save time
#         tasks = [asyncio.create_task(single_attempt()) for _ in range(ATTEMPTS_PER_STEP)]

#         # As each task completes, check if it was successful
#         for completed_task in asyncio.as_completed(tasks):
#             try:
#                 result = await completed_task
#                 if result is not None:
#                     # Cancel remaining tasks
#                     for task in tasks:
#                         task.cancel()
#                     return result
#             except Exception as e:
#                 logger.error(f"Task failed with error: {e}")
#                 continue

#     # If all tasks failed, return error response
#     error_msg = "All concurrent API calls failed"
#     logger.error(error_msg)
#     if is_final_answer:
#         return {
#             "title": "Error",
#             "content": f"Failed to generate final answer. Error: {error_msg}",
#         }
#     else:
#         return {
#             "title": "Error",
#             "content": f"Failed to generate step. Error: {error_msg}",
#             "next_action": "final_answer",
#         }


# async def generate_response(
#     original_messages: list[dict[str, str]], model: str = None, target_uids: list[str] = None, use_miners: bool = True
# ):
#     messages = [
#         {
#             "role": "system",
#             "content": """You are a world-class expert in analytical reasoning and problem-solving. Your task is to break down complex problems through rigorous step-by-step analysis. Present your complete analysis in a single response with numbered steps.

# For each step:
# 1. Provide a brief title
# 2. Give a detailed explanation of your analysis
# 3. Build on previous steps while avoiding redundancy

# Your analysis should include:
# 1. Initial Analysis
#    - Break down the problem into core components
#    - Identify key constraints and requirements
#    - List relevant domain knowledge and principles

# 2. Multiple Perspectives
#    - Examine the problem from different angles
#    - Consider both conventional and unconventional approaches
#    - Identify potential biases in initial assumptions

# 3. Exploration & Validation
#    - Test preliminary conclusions
#    - Apply domain-specific best practices
#    - Document key uncertainties or limitations

# 4. Critical Review & Synthesis
#    - Seek counterarguments to your reasoning
#    - Identify potential failure modes
#    - Combine insights into a comprehensive solution

# End your analysis with a clear final conclusion that:
# 1. States your answer clearly
# 2. Summarizes key supporting evidence
# 3. Acknowledges any remaining uncertainties
# 4. Includes relevant caveats or limitations

# Remember: Quality of reasoning is more important than speed. Take the necessary steps to build a solid analytical foundation.""",
#         }
#     ]
#     messages += original_messages
#     messages += [
#         {
#             "role": "assistant",
#             "content": "I understand. I will now analyze the problem systematically, providing a comprehensive step-by-step analysis leading to a final conclusion.",
#         }
#     ]

#     with Timer() as timer:
#         response = await make_api_call(messages, model=model, use_miners=use_miners, target_uids=target_uids)
#     thinking_time = timer.final_time

#     if response is None:
#         steps = [("Error", "Failed to generate analysis", thinking_time)]
#         yield steps, thinking_time
#         return

#     # Split the response into steps based on "Step X:" pattern
#     step_pattern = re.compile(r"Step \d+:.*?(?=Step \d+:|Final Answer:|$)", re.DOTALL)
#     final_answer_pattern = re.compile(r"Final Answer:.*", re.DOTALL)

#     steps_text = step_pattern.findall(response)
#     final_answer = final_answer_pattern.findall(response)

#     steps = []
#     for step_text in steps_text:
#         # Extract title (first line) and content (rest of the text)
#         step_lines = step_text.strip().split("\n", 1)
#         title = step_lines[0].strip()
#         content = step_lines[1].strip() if len(step_lines) > 1 else ""
#         steps.append((title, content, thinking_time))

#     if final_answer:
#         steps.append(("Final Answer", final_answer[0].replace("Final Answer:", "").strip(), 0))

#     # yield steps, thinking_time
