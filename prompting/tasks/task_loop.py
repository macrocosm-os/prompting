from prompting.base.loop_runner import AsyncLoopRunner
import threading
from prompting.mutable_globals import (
    task_queue,
    scoring_queue,
    TASK_QUEUE_LENGTH_THRESHOLD,
    SCORING_QUEUE_LENGTH_THRESHOLD,
)
from loguru import logger
from prompting.tasks.task_registry import TaskRegistry
from prompting.miner_availability.miner_availability import availability_manager
from prompting.utils.logging import ValidatorLoggingEvent, ErrorLoggingEvent
from pydantic import ConfigDict


class TaskLoop(AsyncLoopRunner):
    is_running: bool = False
    thread: threading.Thread = None
    interval: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def run_step(self) -> ValidatorLoggingEvent | ErrorLoggingEvent | None:
        if len(task_queue) > TASK_QUEUE_LENGTH_THRESHOLD:
            logger.debug("Task queue is full. Skipping task generation.")
            return None
        if len(scoring_queue) > SCORING_QUEUE_LENGTH_THRESHOLD:
            logger.debug("Scoring queue is full. Skipping task generation.")
            return None

        try:
            # Getting task & Dataset
            while True:
                try:
                    task = TaskRegistry.create_random_task_with_dataset()
                    break
                except Exception as ex:
                    logger.exception(ex)

            if len(availability_manager.get_available_miners(task=task, model=task.llm_model_id)) == 0:
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


task_loop = TaskLoop()
