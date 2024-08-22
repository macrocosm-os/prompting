from pydantic import ConfigDict
from loguru import logger
import threading
from prompting.tasks.base_task import BaseTextTask
from prompting.tasks.task_registry import TaskRegistry
from prompting.base.dendrite import DendriteResponseEvent
from prompting.tasks.inference import model_manager
from prompting.utils.logging import RewardLoggingEvent, log_event
import numpy as np
from prompting.datasets.base import DatasetEntry
from dataclasses import dataclass
from prompting.base.loop_runner import AsyncLoopRunner


@dataclass
class ScoringConfig:
    task: BaseTextTask
    response: DendriteResponseEvent
    dataset_entry: DatasetEntry


class ScoringManager(AsyncLoopRunner):
    """The scoring manager maintains a queue of tasks & responses to score and then runs a scoring loop in a background thread.
    This scoring loop will score the responses once the LLM needed is loaded in the model_manager and log the rewards.
    """

    scoring_queue: list[ScoringConfig] = []
    is_running: bool = False
    thread: threading.Thread = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_to_queue(self, task: BaseTextTask, response: DendriteResponseEvent, dataset_entry: DatasetEntry) -> None:
        logger.debug(f"SCORING: Added to queue: {task.task_id}")
        self.scoring_queue.append(ScoringConfig(task=task, response=response, dataset_entry=dataset_entry))

    async def run_step(self) -> RewardLoggingEvent:
        # Only score responses for which the model is loaded
        scorable = [
            scoring_config
            for scoring_config in self.scoring_queue
            if scoring_config.task.model in model_manager.active_models.keys()
        ]
        if len(scorable) == 0:
            logger.debug("Nothing to score. Skipping scoring step.")
            return
        self.scoring_queue.remove(scorable[0])
        scoring_config = scorable.pop(0)

        # here we generate the actual reference
        scoring_config.task.make_reference(
            dataset_entry=scoring_config.dataset_entry,
        )

        # and there we then calculate the reward
        reward_pipeline = TaskRegistry.get_task_reward(scoring_config.task)
        reward_events, penalty_events, rewards = reward_pipeline.apply(
            response_event=scoring_config.response,
            challenge=scoring_config.task.query,
            reference=scoring_config.task.reference,
        )
        best_response = scoring_config.response.completions[np.argmax(rewards)]
        logger.debug(f"SCORING: Scored {scoring_config.task.task_id} with reward {rewards}")
        log_event(
            RewardLoggingEvent(
                best=best_response,
                reward_events=reward_events,
                penalty_events=penalty_events,
                task_id=scoring_config.task.task_id,
            )
        )


scoring_manager = ScoringManager()
