

import copy
import random

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from shared.uids import get_uids
from validator_api.chat_completion import get_response_from_miner, regular_chat_completion


async def mixture_of_miners(body: dict[str, any]) -> tuple | StreamingResponse:
    """Handle chat completion with mixture of miners approach."""
    DEFAULT_SYSTEM_PROMPT = """You have been provided with a set of responses from various open-source models to the latest user query.
    Your task is to synthesize these responses into a single, high-quality and concise response.
    It is crucial to follow the provided instuctions or examples in the given prompt if any, and ensure the answer is in correct and expected format.
    Critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.
    Your response should not simply replicate the given answers but should offer a refined and accurate reply to the instruction.
    Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
    Responses from models:"""

    TASK_SYSTEM_PROMPT = {
        None: DEFAULT_SYSTEM_PROMPT,
    }

    # Get responses from multiple miners
    body_first_step = copy.deepcopy(body)
    body_first_step["stream"] = False

    # Get multiple miners
    miner_uids = get_uids(sampling_mode="top_incentive", k=3)  # Get responses from top 3 miners
    if not miner_uids:
        raise HTTPException(status_code=503, detail="No available miners found")

    # Collect responses from all miners
    responses = []
    for uid in miner_uids:
        try:
            response = await get_response_from_miner(body_first_step, uid)
            responses.append(response)
        except Exception as e:
            logger.error(f"Error getting response from miner {uid}: {e}")
            continue

    if not responses:
        raise HTTPException(status_code=503, detail="Failed to get responses from miners")

    # Extract completions from the responses
    completions = [response[1][0] for response in responses if response and len(response) > 1]

    task_name = body.get("task")
    system_prompt = TASK_SYSTEM_PROMPT.get(task_name, DEFAULT_SYSTEM_PROMPT)

    # Aggregate responses into one system prompt
    agg_system_prompt = system_prompt + "\n" + "\n".join([f"{i+1}. {comp}" for i, comp in enumerate(completions)])

    # Prepare new messages with the aggregated system prompt
    original_messages = body["messages"]
    original_user_messages = [msg for msg in original_messages if msg["role"] != "system"]
    new_messages = [{"role": "system", "content": agg_system_prompt}] + original_user_messages

    # Update the body with the new messages
    final_body = copy.deepcopy(body)
    final_body["messages"] = new_messages

    # Get final response using a random top miner
    final_uid = random.choice(get_uids(sampling_mode="top_incentive", k=100))
    return await regular_chat_completion(final_body, final_uid)
