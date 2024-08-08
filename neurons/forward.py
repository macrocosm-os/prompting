import time
import asyncio
import traceback
from typing import List, Dict, Awaitable
from prompting.base.dendrite import SynapseStreamResult
from prompting.base.protocol import StreamPromptingSynapse
from prompting.utils.misc import async_log, serialize_exception_to_string
from transformers import PreTrainedTokenizerFast as Tokenizer
from prompting.tasks.base_task import BaseTask
from prompting.llms.base_llm import BasePipeline
from loguru import logger


@async_log
async def execute_dendrite_call(dendrite_call):
    responses = await dendrite_call
    return responses


async def process_stream(uid: int, async_iterator: Awaitable, tokenizer: Tokenizer) -> SynapseStreamResult:
    """Process a single response asynchronously."""
    synapse = None  # Initialize chunk with a default value
    exception = None
    accumulated_chunks = []
    accumulated_chunks_timings = []
    accumulated_tokens_per_chunk = []
    start_time = time.time()

    try:
        async for chunk in async_iterator:  # most important loop, as this is where we acquire the final synapse.
            if isinstance(chunk, str):
                accumulated_chunks.append(chunk)
                accumulated_chunks_timings.append(time.time() - start_time)

                tokens_in_chunk = len(tokenizer.tokenize(chunk))
                accumulated_tokens_per_chunk.append(tokens_in_chunk)

                logger.debug(f"\nchunk for uid {uid}: {chunk}")

        # Assuming last chunk of async_iterator holds the last value yielded as a StreamingSynapse
        synapse = chunk
        if synapse is None or not isinstance(synapse, StreamPromptingSynapse):
            raise ValueError(f"Something went wrong with miner uid {uid}, Synapse is not StreamPromptingSynapse.")
    except Exception as e:
        exception = e
        traceback_details = traceback.format_exc()
        logger.error(f"Error in generating reference or handling responses for uid {uid}: {e}\n{traceback_details}")

        failed_synapse = StreamPromptingSynapse(roles=["user"], messages=["failure"], completion="")

        synapse = failed_synapse
    finally:
        return SynapseStreamResult(
            accumulated_chunks=accumulated_chunks,
            accumulated_chunks_timings=accumulated_chunks_timings,
            tokens_per_chunk=accumulated_tokens_per_chunk,
            synapse=synapse,
            uid=uid,
            exception=exception,
        )


@async_log
async def handle_response(stream_results_dict: Dict[int, Awaitable], tokenizer: Tokenizer) -> List[SynapseStreamResult]:
    """The handle_response function is responsible for creating asyncio tasks around acquiring streamed miner chunks
    and processing them asynchronously. It then pairs the results with their original UIDs and returns a list of StreamResults.

    Args:
        responses (Dict[int, Awaitable]): Responses contains awaitables that are used to acquire streamed miner chunks.

    Raises:
        ValueError

    Returns:
        List[StreamResult]: DataClass containing the synapse, exception, and uid
    """
    tasks_with_uid = [
        (uid, stream_results_dict[uid]) for uid, _ in stream_results_dict.items()
    ]  # Pair UIDs with their tasks

    # Start tasks, preserving order and their associated UIDs
    process_stream_tasks = [process_stream(uid, resp, tokenizer) for uid, resp in tasks_with_uid]
    processed_stream_results = await asyncio.gather(*process_stream_tasks, return_exceptions=True)

    return processed_stream_results


@async_log
async def generate_reference(task: BaseTask, pipeline: BasePipeline) -> str:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, task.generate_reference, pipeline)
    return result


def log_stream_results(stream_results: List[SynapseStreamResult]):
    failed_responses = [response for response in stream_results if response.exception is not None]
    empty_responses = [
        response for response in stream_results if response.exception is None and response.synapse.completion == ""
    ]
    non_empty_responses = [
        response for response in stream_results if response.exception is None and response.synapse.completion != ""
    ]

    logger.info(f"Total of non_empty responses: ({len(non_empty_responses)})")
    logger.info(f"Total of empty responses: ({len(empty_responses)})")
    logger.info(f"Total of failed responses: ({len(failed_responses)}):\n {failed_responses}")

    for failed_response in failed_responses:
        formatted_exception = serialize_exception_to_string(failed_response.exception)
        logger.error(f"Failed response for uid {failed_response.uid}: {formatted_exception}")
