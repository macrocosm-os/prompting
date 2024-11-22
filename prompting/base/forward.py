from typing import Awaitable, Dict, List

from loguru import logger

from prompting.base.dendrite import SynapseStreamResult
from prompting.utils.misc import async_log, serialize_exception_to_string


@async_log
async def execute_dendrite_call(dendrite_call):
    responses = await dendrite_call
    return responses

def log_stream_results(stream_results: List[SynapseStreamResult]):
    failed_responses = [
        response for response in stream_results if response.exception is not None or response.completion is None
    ]
    empty_responses = [
        response for response in stream_results if response.exception is None and response.completion == ""
    ]
    non_empty_responses = [
        response for response in stream_results if response.exception is None and response.completion != ""
    ]

    logger.debug(f"Total of non_empty responses: ({len(non_empty_responses)})")
    logger.debug(f"Total of empty responses: ({len(empty_responses)})")
    logger.debug(f"Total of failed responses: ({len(failed_responses)})")

    for failed_response in failed_responses:
        formatted_exception = serialize_exception_to_string(failed_response.exception)
        logger.error(f"Failed response for uid {failed_response.uid}: {formatted_exception}")