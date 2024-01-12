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
    BaseRewardModel,
    RewardEvent,
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

            required_reward_models += SUPPORTED_TASKS[
                task
            ].reward_definition.copy()

        # TODO: Improve readability / design
        # Instantiate only the required reward models
        reward_models = REWARD_MODELS.copy()
        for model in required_reward_models:
            name = model.pop("name")
            reward_models[name] = REWARD_MODELS[name](**model)

        self.reward_models = reward_models
        bt.logging.info(f"Loaded reward models: {self.reward_models.keys()}")

    def reward_responses(self, task, response_event) -> List[RewardEvent]:
        selected_reward_models: Dict[str, BaseRewardModel] = {}

        for reward_definition in task.reward_definition:
            if reward_definition["name"] in self.reward_models:
                reward_model_name = reward_definition["name"]
                selected_reward_models[reward_model_name] = self.reward_models[
                    reward_model_name
                ]

        reward_events = []
        for reward_model in selected_reward_models.values():
            reward_event = reward_model.apply(response_event=response_event)
            reward_events.append(reward_event)

        return reward_events
