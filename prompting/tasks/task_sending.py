# ruff: noqa: E402
import asyncio
import json
import time

from loguru import logger

from prompting import mutable_globals
from shared.dendrite import DendriteResponseEvent
from shared.epistula import query_miners
from shared.loop_runner import AsyncLoopRunner
from prompting.miner_availability.miner_availability import miner_availabilities
from prompting.mutable_globals import scoring_queue
from prompting.rewards.scoring import task_scorer
from prompting.settings import settings
from prompting.tasks.base_task import BaseTextTask
from shared.logging import ErrorLoggingEvent, ValidatorLoggingEvent
from shared.misc import ttl_get_block
from shared.timer import Timer
from typing import List

from loguru import logger

from shared.dendrite import SynapseStreamResult

NEURON_SAMPLE_SIZE = 100


# TODO: do we actually need this logging?
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


async def collect_responses(task: BaseTextTask) -> DendriteResponseEvent | None:
    # Get the list of uids and their axons to query for this step.
    uids = miner_availabilities.get_available_miners(task=task, model=task.llm_model_id, k=NEURON_SAMPLE_SIZE)
    logger.debug(f"🔍 Querying uids: {uids}")
    if len(uids) == 0:
        logger.warning("No available miners. This should already have been caught earlier.")
        return

    body = {
        "seed": task.seed,
        "sampling_parameters": task.sampling_params,
        "task": task.__class__.__name__,
        "model": task.llm_model_id,
        "messages": [
            {"role": "user", "content": task.query},
        ],
    }
    stream_results = await query_miners(uids, body)
    logger.debug(f"🔍 Collected responses from {len(stream_results)} miners")

    log_stream_results(stream_results)

    response_event = DendriteResponseEvent(stream_results=stream_results, uids=uids, timeout=settings.NEURON_TIMEOUT)
    return response_event


class TaskSender(AsyncLoopRunner):
    interval: int = 10
    _lock: asyncio.Lock = asyncio.Lock()
    time_of_block_sync: float | None = None

    @property
    def block(self):
        self.time_of_block_sync = time.time()
        return ttl_get_block()

    @property
    def estimate_block(self):
        """
        Estimate the current block number based on the time since the last block sync.

        Returns:
            Optional[int]: The estimated block number or None if an error occurs.
        """

        if self.time_of_block_sync is None:
            block = self.block
            return block

        # Calculate the block based on the time since the last block
        time_since_last_block = time.time() - self.time_of_block_sync
        # A block happens every 12 seconds
        blocks_since_last_block = time_since_last_block // 12
        estimated_block = int(self.block + blocks_since_last_block)

        return estimated_block

    async def run_step(
        self, k: int = settings.ORGANIC_SAMPLE_SIZE, timeout: float = settings.NEURON_TIMEOUT
    ) -> ValidatorLoggingEvent | ErrorLoggingEvent | None:
        """Executes a single step of the agent, which consists of:
        - Getting a list of uids to query
        - Querying the network
        - Rewarding the network
        - Updating the scores
        - Logging the event
        Args:
            agent (HumanAgent): The agent to run the step for.
            roles (List[str]): The roles for the synapse.
            messages (List[str]): The messages for the synapse.
            k (int): The number of uids to query.
            timeout (float): The timeout for the queries.
            exclude (list, optional): The list of uids to exclude from the query. Defaults to [].
        """
        while len(scoring_queue) > settings.SCORING_QUEUE_LENGTH_THRESHOLD:
            logger.debug("Scoring queue is full. Waiting 1 second...")
            await asyncio.sleep(1)
        while len(mutable_globals.task_queue) == 0:
            logger.warning("No tasks in queue. Waiting 1 second...")
            await asyncio.sleep(1)
        try:
            # get task from the task queue
            mutable_globals.task_queue: list[BaseTextTask]
            task = mutable_globals.task_queue.pop(0)

            # send the task to the miners and collect the responses
            with Timer() as timer:
                response_event = await collect_responses(task=task)
            if response_event is None:
                logger.warning("No response event collected. This should not be happening.")
                return
            logger.debug(f"Collected responses in {timer.final_time:.2f} seconds")

            # scoring_manager will score the responses as and when the correct model is loaded
            task_scorer.add_to_queue(
                task=task,
                response=response_event,
                dataset_entry=task.dataset_entry,
                block=self.estimate_block,
                step=self.step,
                task_id=task.task_id,
            )

            # Log the step event.
            return ValidatorLoggingEvent(
                block=self.estimate_block,
                step=self.step,
                step_time=timer.final_time,
                response_event=response_event,
                task_id=task.task_id,
            )

        except Exception as ex:
            logger.exception(ex)
            return ErrorLoggingEvent(
                error=str(ex),
            )


task_sender = TaskSender()
