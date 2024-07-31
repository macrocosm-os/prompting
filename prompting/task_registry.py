# from prompting.tasks.date_qa import DateQuestionAnsweringTask
from prompting.tasks.task import BaseTask, BaseRewardModel
from prompting.tasks.summarization import SummarizationTask, SummarizationRewardConfig
from prompting.tasks.qa import QuestionAnsweringTask, QARewardConfig

from prompting.tools.datasets.wiki import WikiDataset
from prompting.tools.datasets.base import BaseDataset
from pydantic import BaseModel
import random
from typing import ClassVar
import bittensor as bt


class TaskConfig(BaseModel):
    task: ClassVar[BaseTask]
    probability: ClassVar[float]
    datasets: ClassVar[list[BaseDataset]]
    reward_model: ClassVar[BaseRewardModel]


class TaskRegistry(BaseModel):
    tasks: ClassVar[list[TaskConfig]] = [
        TaskConfig(task=QuestionAnsweringTask, probability=0.6, datasets=[WikiDataset], reward_model=QARewardConfig),
        TaskConfig(
            task=SummarizationTask, probability=0.4, datasets=[WikiDataset], reward_model=SummarizationRewardConfig
        ),
        # TaskConfig(task=DateQuestionAnsweringTask, probability=0.2, datasets=[WikiDateDataset])
    ]

    @classmethod
    def random(cls) -> TaskConfig:
        probabilities = [task.probability for task in cls.tasks]
        selected_task = random.choices(cls.tasks, probabilities)[0]
        return selected_task

    @classmethod
    def get_task_datasets(cls, task: BaseTask):
        try:
            return [t.datasets for t in cls.tasks if isinstance(task, t.__class__)][0]
        except Exception:
            bt.logging.error("Tried accessing non-registered task")
            return []

    @classmethod
    def get_random_task_dataset(cls, task: BaseTask) -> BaseDataset:
        return random.choice(cls.get_task_datasets(task))

    @classmethod
    def get_task_reward(cls, task: BaseTask) -> BaseRewardModel:
        try:
            return [t.reward_model for t in cls.tasks if isinstance(task, t.__class__)][0]
        except Exception:
            bt.logging.error("Tried accessing non-registered task")
            return []

    @classmethod
    def create_random_task(cls, llm_pipeline) -> BaseTask:
        task_config = cls.random()
        dataset = cls.get_random_task_dataset(task_config.task)
        return task_config.task(
            llm_pipeline=llm_pipeline, context=dataset.next(), reward_config=task_config.reward_model()
        )
