import asyncio
import copy
import random

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from shared.uids import get_uids
from validator_api.chat_completion import get_response_from_miner, regular_chat_completion


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

NUM_MIXTURE_MINERS = 5
TOP_INCENTIVE_POOL = 100



async def get_miner_response(body: dict, uid: str) -> tuple | None:
    """Get response from a single miner with error handling."""
    try:
        return await get_response_from_miner(body, uid)
    except Exception as e:
        logger.error(f"Error getting response from miner {uid}: {e}")
        return None


async def mixture_of_miners(body: dict[str, any]) -> tuple | StreamingResponse:
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
    miner_uids = get_uids(sampling_mode="top_incentive", k=NUM_MIXTURE_MINERS)
    if not miner_uids:
        raise HTTPException(status_code=503, detail="No available miners found")

    # Concurrently collect responses from all miners.
    miner_tasks = [get_miner_response(body_first_step, uid) for uid in miner_uids]
    responses = await asyncio.gather(*miner_tasks)
    
    # Filter out None responses (failed requests).
    valid_responses = [r for r in responses if r is not None]
    
    if not valid_responses:
        raise HTTPException(status_code=503, detail="Failed to get responses from miners")

    # Extract completions from the responses.
    completions = [response[1][0] for response in valid_responses if response and len(response) > 1]
    
    task_name = body.get("task")
    system_prompt = TASK_SYSTEM_PROMPT.get(task_name, DEFAULT_SYSTEM_PROMPT)
    
    # Aggregate responses into one system prompt.
    agg_system_prompt = system_prompt + "\n" + "\n".join([f"{i+1}. {comp}" for i, comp in enumerate(completions)])
    
    # Prepare new messages with the aggregated system prompt.
    new_messages = [{"role": "system", "content": agg_system_prompt}]
    new_messages.extend([msg for msg in body["messages"] if msg["role"] != "system"])
    
    # Update the body with the new messages.
    final_body = copy.deepcopy(body)
    final_body["messages"] = new_messages
    
    # Get final response using a random top miner.
    final_uid = random.choice(get_uids(sampling_mode="top_incentive", k=TOP_INCENTIVE_POOL))
    return await regular_chat_completion(final_body, final_uid)
