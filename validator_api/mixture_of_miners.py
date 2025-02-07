import asyncio
import copy
import math
import random

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from shared.uids import get_uids
from validator_api.chat_completion import chat_completion, get_response_from_miner

DEFAULT_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query.
Your task is to synthesize these responses into a single, high-quality and concise response.
It is crucial to follow the provided instuctions or examples in the given prompt if any, and ensure the answer is in correct and expected format.
Critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.
Your response should not simply replicate the given answers but should offer a refined and accurate reply to the instruction.
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models:"""

TASK_SYSTEM_PROMPT = {
    None: DEFAULT_SYSTEM_PROMPT,
    # Add more task-specific system prompts here.
}

NUM_MIXTURE_MINERS = 8
TOP_INCENTIVE_POOL = 100


async def get_miner_response(body: dict, uid: str, timeout_seconds: int) -> tuple | None:
    """Get response from a single miner with error handling."""
    try:
        return await get_response_from_miner(body, uid, timeout_seconds=timeout_seconds)
    except Exception as e:
        logger.error(f"Error getting response from miner {uid}: {e}")
        return None


async def mixture_of_miners(body: dict[str, any], uids: list[int]) -> tuple | StreamingResponse:
    """Handle chat completion with mixture of miners approach.

    Based on Mixture-of-Agents Enhances Large Language Model Capabilities, 2024, Wang et al.:
        https://arxiv.org/abs/2406.04692

    Args:
        body: Query parameters:
            messages: User prompt.
            stream: If True, stream the response.
            model: Optional model used for inference, SharedSettings.LLM_MODEL is used by default.
            task: Optional task, see prompting/tasks/task_registry.py, InferenceTask is used by default.
    """
    body_first_step = copy.deepcopy(body)
    body_first_step["stream"] = False

    # Get multiple miners
    if not uids:
        uids = get_uids(sampling_mode="top_incentive", k=NUM_MIXTURE_MINERS)
    if len(uids) == 0:
        raise HTTPException(status_code=503, detail="No available miners found")

    # Concurrently collect responses from all miners.
    timeout_seconds = max(
        30, max(0, math.floor(math.log2(body["sampling_parameters"].get("max_new_tokens", 256) / 256))) * 10 + 30
    )
    miner_tasks = [get_miner_response(body_first_step, uid, timeout_seconds=timeout_seconds) for uid in uids]
    responses = await asyncio.gather(*miner_tasks)

    # Filter out None responses (failed requests).
    valid_responses = [r for r in responses if r is not None]

    if not valid_responses:
        raise HTTPException(status_code=503, detail="Failed to get responses from miners")

    # Extract completions from the responses.
    completions = ["".join(response[1]) for response in valid_responses if response and len(response) > 1]

    task_name = body.get("task")
    system_prompt = TASK_SYSTEM_PROMPT.get(task_name, DEFAULT_SYSTEM_PROMPT)
    logger.debug(f"Using Mixture of Miners with {len(completions)} miners")

    new_messages = body["messages"] + [
        {
            "role": "assistant",
            "content": "I have received the following responses from various models:\n"
            + "\n".join([f"{i+1}. {comp}" for i, comp in enumerate(completions)])
            + "Now I will synthesize them into a single, high-quality and concise response to the user's query.",
        },
    ]

    # Update the body with the new messages.
    final_body = copy.deepcopy(body)
    final_body["messages"] = new_messages

    # Get final response using a random top miner.
    final_uid = random.choice(get_uids(sampling_mode="top_incentive", k=TOP_INCENTIVE_POOL))
    return await chat_completion(final_body, uids=[int(final_uid)])
