import bittensor as bt
from typing import List, Dict
from tasks import DebuggingTask, SummarizationTask, QuestionAnsweringTask, MathTask, DateQuestionAnsweringTask
from rewards import BaseRewardModel, RewardEvent, RougeRewardModel, DiffRewardModel ,RelevanceRewardModel


SUPPORTED_TASKS = {
    'debugging': DebuggingTask,
    'summarization': SummarizationTask,
    'qa': QuestionAnsweringTask,
    'math': MathTask,
    'date_qa': DateQuestionAnsweringTask
}


REWARD_MODELS = {
    'rouge': RougeRewardModel,
    'relevance': RelevanceRewardModel,
    'diff': DiffRewardModel
}


class RewardPipeline:
    def __init__(self, selected_tasks: List[str]):
        self.selected_tasks = selected_tasks
        self.load_pipeline()

    def load_pipeline(self):
        required_reward_models = []

        for task in self.selected_tasks:
            if task not in SUPPORTED_TASKS:
                raise ValueError(f'Task {task} not supported. Please choose from {SUPPORTED_TASKS.keys()}')

            required_reward_models += SUPPORTED_TASKS[task].reward_definition

        # TODO: Improve readability / design
        # Instantiate only the required reward models
        reward_models = REWARD_MODELS.copy()
        for model in required_reward_models:
            name = model['name']
            reward_models[name] = REWARD_MODELS[name](**model)

        self.reward_models = reward_models
        bt.logging.info(f'Loaded reward models: {self.reward_models.keys()}')


    def reward_responses(self, task, response_event: NetworkResponseEvent) -> List[RewardEvent]:
        selected_reward_models: Dict[str, BaseRewardModel] = {}

        for reward_definition in task.reward_definition:
            if reward_definition['name'] in self.reward_models:
                reward_model_name = reward_definition['name']
                selected_reward_models[reward_model_name]=self.reward_models[reward_model_name]


        reward_events = []
        for reward_model in selected_reward_models.values():
            reward_event = reward_model.apply(response_event=response_event)
            reward_events.append(reward_event)


        return reward_events


def get_rewards(self, task, rewards_events: List[RewardEvent]) -> torch.FloatTensor:
    # TODO: How would using the Agent as a reward model fit into this flow?
    # Compute the rewards for the responses given the prompt
    # Creates a dict with the uids as keys and the final rewards as values
    uids_final_rewards = {}

    for task_reward_definition in task.reward_definition:
        # Gets appropriate reward event for the reward model defined in the task
        reward_event = next((event for event in rewards_events if task_reward_definition['name'] == event.model), None)

        if reward_event.model_type == RewardModelTypeEnum.WEIGHTED_REWARD:
            for uid, reward in zip(reward_event.uids, reward_event.rewards):
                # Sets uid as int instead of tensor
                uid = uid.item()
                # Multiplies the reward by the weight defined in the task
                final_rewards = task_reward_definition['weight'] * reward
                # Adds the reward to the uid's final reward
                uid_reward = uids_final_rewards.get(uid, 0)
                uids_final_rewards[uid] = uid_reward + final_rewards

        elif reward_event.model_type == RewardModelTypeEnum.FILTER_REWARD:
            ...
        elif reward_event.model_type == RewardModelTypeEnum.PENALTY:
            ...
        else:
            raise ValueError(f'Reward model type {reward_event.model_type} not supported.')

    final_rewards = torch.tensor(list(uids_final_rewards.values())).to(self.device)

    return final_rewards