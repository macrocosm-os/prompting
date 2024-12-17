import asyncio
import threading

from loguru import logger
from pydantic import ConfigDict

from prompting.miner_availability.miner_availability import miner_availabilities
from prompting.mutable_globals import scoring_queue, task_queue
from prompting.tasks.task_registry import TaskRegistry
from shared.logging import ErrorLoggingEvent, ValidatorLoggingEvent
from shared.loop_runner import AsyncLoopRunner
from shared.settings import settings

RETRIES = 3


class TaskLoop(AsyncLoopRunner):
    is_running: bool = False
    thread: threading.Thread = None
    interval: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def run_step(self) -> ValidatorLoggingEvent | ErrorLoggingEvent | None:
        if len(task_queue) > settings.TASK_QUEUE_LENGTH_THRESHOLD:
            logger.debug("Task queue is full. Skipping task generation.")
            return None
        if len(scoring_queue) > settings.SCORING_QUEUE_LENGTH_THRESHOLD:
            logger.debug("Scoring queue is full. Skipping task generation.")
            return None
        await asyncio.sleep(0.1)
        try:
            # Getting task & Dataset
            for i in range(RETRIES):
                try:
                    logger.debug(f"Retry: {i}")
                    task = TaskRegistry.create_random_task_with_dataset()
                    break
                except Exception as ex:
                    logger.exception(ex)
                await asyncio.sleep(0.1)

            await asyncio.sleep(0.1)
            if len(miner_availabilities.get_available_miners(task=task, model=task.llm_model_id)) == 0:
                logger.debug(
                    f"No available miners for Task: {task.__class__.__name__} and Model ID: {task.llm_model_id}. Skipping step."
                )
                return None

            if not (dataset_entry := task.dataset_entry):
                logger.warning(f"Dataset for task {task.__class__.__name__} returned None. Skipping step.")
                return None

            # Generate the query and reference for the task
            if not task.query:
                logger.debug(f"Generating query for task: {task.__class__.__name__}.")
                task.make_query(dataset_entry=dataset_entry)
            task_queue.append(task)
        except Exception as ex:
            logger.exception(ex)
            return None
        await asyncio.sleep(0.01)


task_loop = TaskLoop()
