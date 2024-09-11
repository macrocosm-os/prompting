from pydantic import ConfigDict
from loguru import logger
import threading
from prompting.tasks.base_task import BaseTextTask
from prompting.tasks.task_registry import TaskRegistry
from prompting.base.dendrite import DendriteResponseEvent
from prompting.llms.model_manager import model_manager, model_scheduler
from prompting.utils.logging import RewardLoggingEvent, log_event
import numpy as np
from prompting.datasets.base import DatasetEntry
from dataclasses import dataclass
from prompting.base.loop_runner import AsyncLoopRunner
import asyncio
from prompting.mutable_globals import scoring_queue, rewards_and_uids


@dataclass
class ScoringConfig:
    task: BaseTextTask
    response: DendriteResponseEvent
    dataset_entry: DatasetEntry
    block: int
    step: int
    task_id: str

class TaskScorer(AsyncLoopRunner):
    """The scoring manager maintains a queue of tasks & responses to score and then runs a scoring loop in a background thread.
    This scoring loop will score the responses once the LLM needed is loaded in the model_manager and log the rewards.
    """

    is_running: bool = False
    thread: threading.Thread = None
    interval: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_to_queue(self, task: BaseTextTask, response: DendriteResponseEvent, dataset_entry: DatasetEntry, block: int, step: int, task_id: str) -> None:
        logger.debug(f"SCORING: Added to queue: {task.__class__.__name__} {task.task_id}")
        scoring_queue.append(ScoringConfig(task=task, response=response, dataset_entry=dataset_entry, block=block, step=step, task_id=task_id))

    async def run_step(self) -> RewardLoggingEvent:
        # Only score responses for which the model is loaded
        scorable = [
            scoring_config
            for scoring_config in scoring_queue
            if (scoring_config.task.llm_model in model_manager.active_models.keys())
            or (scoring_config.task.llm_model is None)
        ]
        if len(scorable) == 0:
            logger.debug("Nothing to score. Skipping scoring step.")
            # Run a model_scheduler step to load a new model as there are no more tasks to be scored
            await model_scheduler.run_step()
            await asyncio.sleep(5)
            return
        scoring_queue.remove(scorable[0])
        scoring_config: ScoringConfig = scorable.pop(0)

        # here we generate the actual reference
        scoring_config.task.make_reference(
            dataset_entry=scoring_config.dataset_entry,
        )

        # and there we then calculate the reward
        reward_pipeline = TaskRegistry.get_task_reward(scoring_config.task)
        logger.debug(
            f"""{len(scoring_config.response.completions)} completions to score for task {scoring_config.task}
            COMPLETIONS: {scoring_config.response.completions}"""
        )
        reward_events, penalty_events, rewards = reward_pipeline.apply(
            response_event=scoring_config.response,
            challenge=scoring_config.task.query,
            reference=scoring_config.task.reference,
            model_id=scoring_config.task.llm_model,
        )
        best_response = (
            scoring_config.response.completions[np.argmax(rewards)]
            if (rewards is not None and len(rewards) > 0)
            else None
        )
        logger.debug(
            f"SCORING: Scored {scoring_config.task.__class__.__name__} {scoring_config.task.task_id} with reward {rewards}"
        )
        log_event(
            RewardLoggingEvent(
                best=best_response,
                response_event=scoring_config.response,
                reward_events=reward_events,
                penalty_events=penalty_events,
                reference=scoring_config.task.reference,
                challenge=scoring_config.task.query,
                task=scoring_config.task.name,
                rewards=rewards,
                block=scoring_config.block,
                step=scoring_config.step,
                task_id=scoring_config.task_id,
            )
        )
        logger.info("Adding scores to rewards_and_uids")
        rewards_and_uids.append((scoring_config.response.uids, rewards))

class WeightSetter(AsyncLoopRunner):
    pass

task_scorer = TaskScorer()
