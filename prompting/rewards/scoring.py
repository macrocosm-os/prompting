import asyncio
import threading

from loguru import logger
from pydantic import ConfigDict

from prompting.llms.model_manager import model_manager, model_scheduler
from prompting.rewards.scoring_config import ScoringConfig
from prompting.tasks.base_task import BaseTextTask
from prompting.tasks.task_registry import TaskRegistry
from shared.base import DatasetEntry
from shared.dendrite import DendriteResponseEvent
from shared.logging import RewardLoggingEvent, log_event
from shared.loop_runner import AsyncLoopRunner


class TaskScorer(AsyncLoopRunner):
    """The scoring manager maintains a queue of tasks & responses to score and then runs a scoring loop in a background thread.
    This scoring loop will score the responses once the LLM needed is loaded in the model_manager and log the rewards.
    """

    is_running: bool = False
    thread: threading.Thread = None
    interval: int = 10
    scoring_queue: list | None = None
    reward_events: list | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def start(self, scoring_queue, reward_events, name: str | None = None):
        self.scoring_queue = scoring_queue
        self.reward_events = reward_events
        return await super().start(name=name)

    def add_to_queue(
        self,
        task: BaseTextTask,
        response: DendriteResponseEvent,
        dataset_entry: DatasetEntry,
        block: int,
        step: int,
        task_id: str,
    ) -> None:
        self.scoring_queue.append(
            ScoringConfig(
                task=task,
                response=response,
                dataset_entry=dataset_entry,
                block=block,
                step=step,
                task_id=task_id,
            )
        )

    async def run_step(self) -> RewardLoggingEvent:
        await asyncio.sleep(0.1)
        # Only score responses for which the model is loaded
        scorable = [
            scoring_config
            for scoring_config in self.scoring_queue
            if (scoring_config.task.llm_model in model_manager.active_models.keys())
            or (scoring_config.task.llm_model is None)
        ]
        if len(scorable) == 0:
            # Run a model_scheduler step to load a new model as there are no more tasks to be scored
            if len(self.scoring_queue) > 0:
                await model_scheduler.run_step()
            return
        self.scoring_queue.remove(scorable[0])
        scoring_config: ScoringConfig = scorable.pop(0)

        # here we generate the actual reference
        scoring_config.task.make_reference(
            dataset_entry=scoring_config.dataset_entry,
        )

        # and there we then calculate the reward
        reward_pipeline = TaskRegistry.get_task_reward(scoring_config.task)
        reward_events = reward_pipeline.apply(
            response_event=scoring_config.response,
            challenge=scoring_config.task.query,
            reference=scoring_config.task.reference,
            model_id=scoring_config.task.llm_model,
            task=scoring_config.task,
        )
        self.reward_events.append(reward_events)
        if scoring_config.task.organic:
            self.reward_events.append(reward_events) # Add the organic a second time, doubling the weight of the organic
        logger.debug(
            f"Scored {scoring_config.task.__class__.__name__} {scoring_config.task.task_id} with model "
            f"{scoring_config.task.llm_model_id}"
        )
        if not scoring_config.task.organic:
            log_event(
                RewardLoggingEvent(
                    response_event=scoring_config.response,
                    reward_events=reward_events,
                    reference=scoring_config.task.reference,
                    challenge=scoring_config.task.query,
                    task=scoring_config.task.name,
                    block=scoring_config.block,
                    step=scoring_config.step,
                    task_id=scoring_config.task_id,
                    task_dict=scoring_config.task.model_dump(),
                    source=scoring_config.dataset_entry.source,
                )
            )
        await asyncio.sleep(0.01)


class WeightSetter(AsyncLoopRunner):
    pass


task_scorer = TaskScorer()
