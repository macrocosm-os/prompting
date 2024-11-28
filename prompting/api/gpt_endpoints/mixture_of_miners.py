import copy

from prompting.api.gpt_endpoints.process_completions import process_completions

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


async def mixture_of_miners(
    body: dict[str, any],
):
    body_1st_step = copy.deepcopy(body)
    body_1st_step["stream"] = False

    # First step: Get initial responses from miners.
    responses = await process_completions(body_1st_step)

    # Extract completions from the responses.
    completions = ["".join(res["accumulated_chunks"]) for res in responses]

    task_name = body.get("task")
    system_prompt = TASK_SYSTEM_PROMPT.get(task_name, DEFAULT_SYSTEM_PROMPT)

    # Aggregate responses into one system prompt.
    agg_system_prompt = system_prompt + "\n" + "\n".join([f"{i+1}. {comp}" for i, comp in enumerate(completions)])

    # Prepare new messages with the aggregated system prompt.
    original_messages = body["messages"]
    original_user_messages = [msg for msg in original_messages if msg["role"] != "system"]
    new_messages = [{"role": "system", "content": agg_system_prompt}] + original_user_messages

    # Update the body with the new messages.
    body["messages"] = new_messages

    # Second step: Get the final response using the aggregated system prompt.
    return await process_completions(body)
