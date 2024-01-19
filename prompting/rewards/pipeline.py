from typing import List

from prompting.tasks import (
    DebuggingTask,
    SummarizationTask,
    QuestionAnsweringTask,
    MathTask,
    DateQuestionAnsweringTask,
)
from prompting.rewards import (
    BaseRewardModel,
    RougeRewardModel,
    DiffRewardModel,
    RelevanceRewardModel,
    FloatDiffModel,
    DateRewardModel,
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
    'float_diff': FloatDiffModel,
    'date': DateRewardModel,
}



class RewardPipeline:
    def __init__(self, selected_tasks: List[str], device):
        self.selected_tasks = selected_tasks
        self.device = device
        self.load_pipeline()

    def __getitem__(self, __key: str) -> BaseRewardModel:
        return self.reward_models.get(__key)

    def get(self, __key: str) -> BaseRewardModel:
        return self.reward_models.get(__key)

    def __repr__(self):
        return f'RewardPipeline({self.reward_models})'

    def load_pipeline(self):
        """Dynamically loads the reward models required by the selected tasks so that we only use the necessary resources."""
        active_reward_models = []

        for task in self.selected_tasks:
            if task not in SUPPORTED_TASKS:
                raise ValueError(
                    f"Task {task} not supported. Please choose from {SUPPORTED_TASKS.keys()}"
                )
            active_reward_models += SUPPORTED_TASKS[task].reward_definition
            active_reward_models += SUPPORTED_TASKS[task].penalty_definition

        # Instantiate only the required reward models
        reward_models = {}
        for model in active_reward_models:
            name = model.get("name")
            if not name:
                raise ValueError(f"Reward model {model} does not have a name. ")
            if name not in REWARD_MODELS:
                raise ValueError(
                    f"Reward model {name} not supported. Please choose from {REWARD_MODELS.keys()}"
                )
            elif name in reward_models: # Prevents duplicate reward models
                continue

            cls = REWARD_MODELS[name]

            params = {k: v for k, v in model.items() if k not in ["name", "weight"]}
            reward_models[name] = cls(device=self.device, **params)

        self.reward_models = reward_models

