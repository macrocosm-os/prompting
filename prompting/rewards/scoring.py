from pydantic import BaseModel, ConfigDict
from loguru import logger
import threading
from prompting.tasks.base_task import BaseTextTask
import bittensor as bt
from prompting.tasks.task_registry import TaskRegistry
import time
from prompting.base.dendrite import DendriteResponseEvent
from prompting.tasks.inference import model_manager
from prompting.utils.logging import RewardLoggingEvent, log_event
import numpy as np
from prompting.datasets.base import Context


class ScoringConfig(BaseModel):
    task: BaseTextTask
    response: DendriteResponseEvent
    dataset_entry: Context


class ScoringManager(BaseModel):
    """The scoring manager maintains a queue of tasks & responses to score and then runs a scoring loop in a background thread.
    This scoring loop will score the responses once the LLM needed is loaded in the model_manager and log the rewards.
    """

    scoring_queue: list[ScoringConfig] = []
    is_running: bool = False
    thread: threading.Thread = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_to_queue(self, task: BaseTextTask, response: bt.Synapse, dataset_entry: Context) -> None:
        self.scoring_queue.append(ScoringConfig(task=task, response=response, dataset_entry=dataset_entry))

    def scoring_step(self) -> RewardLoggingEvent:
        # Only score responses for which the model is loaded
        scorable = [
            scoring_config
            for scoring_config in self.scoring_queue
            if scoring_config.task.model in model_manager.active_models.keys()
        ]
        best_response: str = ""
        reward_events = []
        penalty_events = []
        if len(scorable) > 0:
            self.scoring_queue.remove(scorable[0])
            scoring_config = scorable.pop(0)

            # here we generate the actual reference
            scoring_config.task.make_reference(
                context=scoring_config.dataset_entry,
            )

            # and there we then calculate the reward
            reward_pipeline = TaskRegistry.get_task_reward(scoring_config.task)
            reward_events, penalty_events, rewards = reward_pipeline.apply(
                response_event=scoring_config.response,
                challenge=scoring_config.task.query,
                reference=scoring_config.task.reference,
            )
            best_response = scoring_config.response.completions[np.argmax(rewards)]

        else:
            time.sleep(1)

        # Log the step event.
        log_event(
            RewardLoggingEvent(
                best=best_response,
                reward_events=reward_events,
                penalty_events=penalty_events,
                task_id=scoring_config.task.task_id,
            )
        )

    def run_scoring_loop(self) -> None:
        while True:
            self.scoring_step()
            logger.debug("Scoring step completed.")

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            self.thread = threading.Thread(target=self.run_scoring_loop, daemon=True)
            self.thread.start()
            self.is_running = True


scoring_manager = ScoringManager()
scoring_manager.run_in_background_thread()
