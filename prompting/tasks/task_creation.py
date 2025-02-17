import asyncio
import threading

from loguru import logger
from pydantic import ConfigDict

from prompting.miner_availability.miner_availability import miner_availabilities
from prompting.tasks.task_registry import TaskRegistry
from shared import settings

# from shared.logging import ErrorLoggingEvent, ValidatorLoggingEvent
from shared.loop_runner import AsyncLoopRunner

shared_settings = settings.shared_settings

RETRIES = 3


class TaskLoop(AsyncLoopRunner):
    is_running: bool = False
    thread: threading.Thread = None
    interval: int = 10
    task_queue: list | None = []
    scoring_queue: list | None = []
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def start(self, task_queue, scoring_queue):
        self.task_queue = task_queue
        self.scoring_queue = scoring_queue
        await super().start()

    async def run_step(self):
        if len(self.task_queue) > shared_settings.TASK_QUEUE_LENGTH_THRESHOLD:
            return None
        if len(self.scoring_queue) > shared_settings.SCORING_QUEUE_LENGTH_THRESHOLD:
            return None
        await asyncio.sleep(0.1)
        try:
            task = None
            # Getting task and dataset
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

            logger.debug(f"Appending task: {task.__class__.__name__} to task queue.")
            self.task_queue.append(task)
        except Exception as ex:
            logger.exception(ex)
            return None
        await asyncio.sleep(0.01)


task_loop = TaskLoop()
