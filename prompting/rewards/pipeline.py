import bittensor as bt
from typing import List, Dict

from prompting.tasks import (
    DebuggingTask,
    SummarizationTask,
    QuestionAnsweringTask,
    MathTask,
    DateQuestionAnsweringTask,
)
from prompting.rewards import (
    RougeRewardModel,
    DiffRewardModel,
    RelevanceRewardModel,
)


SUPPORTED_TASKS = {
    "debugging": DebuggingTask,
    "summarization": SummarizationTask,
    "qa": QuestionAnsweringTask,
    "math": MathTask,
    "date_qa": DateQuestionAnsweringTask,
}


REWARD_MODELS = {
    "rouge": RougeRewardModel,
    "relevance": RelevanceRewardModel,
    "diff": DiffRewardModel,
}



class RewardPipeline:
    def __init__(self, selected_tasks: List[str]):
        self.selected_tasks = selected_tasks
        self.load_pipeline()

    def load_pipeline(self):
        """Dynamically loads the reward models required by the selected tasks so that we only use the necessary resources."""
        required_reward_models = []

        for task in self.selected_tasks:
            if task not in SUPPORTED_TASKS:
                raise ValueError(
                    f"Task {task} not supported. Please choose from {SUPPORTED_TASKS.keys()}"
                )
            required_reward_models += SUPPORTED_TASKS[task].reward_definition.copy()

        # Instantiate only the required reward models
        reward_models = REWARD_MODELS.copy()
        for model in required_reward_models:
            name = model.pop("name")
            weight = model.pop("weight")
            reward_models[name] = REWARD_MODELS[name](**model)

        self.reward_models = reward_models
        bt.logging.info(f"Loaded reward models: {self.reward_models.keys()}")

