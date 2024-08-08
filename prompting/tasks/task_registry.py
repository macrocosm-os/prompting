# from prompting.tasks.date_qa import DateQuestionAnsweringTask
from prompting.tasks.base_task import BaseTask
from prompting.rewards.reward import BaseRewardConfig
from prompting.tasks.date_qa import DateQuestionAnsweringTask, DateQARewardConfig
from prompting.tasks.qa import QuestionAnsweringTask, QARewardConfig

from prompting.datasets.wiki import WikiDataset, WikiDateDataset
from prompting.datasets.base import BaseDataset
from pydantic import BaseModel, ConfigDict
import random
from typing import ClassVar
from loguru import logger


class TaskConfig(BaseModel):
    task: BaseTask.__class__
    probability: float
    datasets: list[BaseDataset.__class__]
    reward_model: BaseRewardConfig.__class__

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TaskRegistry(BaseModel):
    tasks: ClassVar[list[TaskConfig]] = [
        TaskConfig(task=QuestionAnsweringTask, probability=0.2, datasets=[WikiDataset], reward_model=QARewardConfig),
        # TaskConfig(
        #     task=SummarizationTask, probability=0.4, datasets=[WikiDataset], reward_model=SummarizationRewardConfig
        # ),
        TaskConfig(
            task=DateQuestionAnsweringTask, probability=0.8, datasets=[WikiDateDataset], reward_model=DateQARewardConfig
        ),
    ]

    @classmethod
    def random(cls) -> TaskConfig:
        probabilities = [task.probability for task in cls.tasks]
        selected_task = random.choices(cls.tasks, probabilities)[0]
        return selected_task

    @classmethod
    def get_task_datasets(cls, task: BaseTask.__class__) -> BaseDataset.__class__:
        try:
            return [t.datasets for t in cls.tasks if task is t.task][0]
        except Exception:
            logger.error("Tried accessing non-registered task")
            return []

    @classmethod
    def get_random_task_dataset(cls, task: BaseTask.__class__) -> BaseDataset.__class__:
        return random.choice(cls.get_task_datasets(task))

    @classmethod
    def get_task_reward(cls, task: BaseTask | BaseTask.__class__) -> BaseRewardConfig.__class__:
        task_class = task.__class__ if isinstance(task, BaseTask) else task
        try:
            return [t.reward_model for t in cls.tasks if task_class is t.task][0]
        except Exception:
            logger.error("Tried accessing non-registered task")
            return []

    @classmethod
    def create_random_task_with_dataset(cls) -> tuple[BaseTask.__class__, BaseDataset]:
        task_config = cls.random()
        dataset = cls.get_random_task_dataset(task_config.task)
        return task_config.task, dataset()
