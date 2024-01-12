import torch
import time
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class RewardModelTypeEnum(Enum):
    WEIGHTED_REWARD = "weighted_reward"
    FILTER_REWARD = "filter_reward"
    PENALTY = "penalty"


# Move code out of here
@dataclass
class RewardEvent:
    rewards: torch.FloatTensor
    timings: List[float]
    model: str
    model_type: RewardModelTypeEnum
    batch_time: float
    extra_info: dict
    uids: List[float]

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> dict:
        return {
            f"{self.model}_model_raw_rewards": self.rewards,
            f"{self.model}_model_timings": [
                round(num, 5) for num in self.timings
            ]
            if self.timings
            else None,
            f"{self.model}_model_batch_time": self.batch_time,
            f"{self.model}_model_extra_info": self.extra_info,
        }


class RewardResult:
    def __init__(self, reward_pipeline, task, response_event):
        self.reward_pipeline = reward_pipeline
        self.task = task
        self.response_event = response_event

    def get_rewards(
        self, task, rewards_events: List[RewardEvent]
    ) -> torch.FloatTensor:
        # TODO: How would using the Agent as a reward model fit into this flow?
        # Compute the rewards for the responses given the prompt
        # Creates a dict with the uids as keys and the final rewards as values
        uids_final_rewards = {}

        for task_reward_definition in task.reward_definition:
            # Gets appropriate reward event for the reward model defined in the task
            reward_event = next(
                (
                    event
                    for event in rewards_events
                    if task_reward_definition["name"] == event.model
                ),
                None,
            )

            if reward_event.model_type == RewardModelTypeEnum.WEIGHTED_REWARD:
                for uid, reward in zip(reward_event.uids, reward_event.rewards):
                    # Sets uid as int instead of tensor
                    uid = uid.item()
                    # Multiplies the reward by the weight defined in the task
                    final_rewards = task_reward_definition["weight"] * reward
                    # Adds the reward to the uid's final reward
                    uid_reward = uids_final_rewards.get(uid, 0)
                    uids_final_rewards[uid] = uid_reward + final_rewards

            elif reward_event.model_type == RewardModelTypeEnum.FILTER_REWARD:
                ...
            elif reward_event.model_type == RewardModelTypeEnum.PENALTY:
                ...
            else:
                raise ValueError(
                    f"Reward model type {reward_event.model_type} not supported."
                )

        final_rewards = torch.tensor(list(uids_final_rewards.values())).to(
            self.device
        )

        return final_rewards


@dataclass
class BatchRewardOutput:
    rewards: torch.FloatTensor
    timings: List[float]
    extra_info: dict


class BaseRewardModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_type(self) -> RewardModelTypeEnum:
        ...

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reward(
        self, reference: str, completions: List[str]
    ) -> BatchRewardOutput:
        pass

    def apply(self, response_event) -> RewardEvent:
        t0 = time.time()
        batch_rewards_output = self.reward(
            response_event.reference, response_event.completions
        )
        batch_rewards_time = time.time() - t0

        reward_event = RewardEvent(
            rewards=batch_rewards_output.rewards.cpu().tolist(),
            model_type=self.model_type,
            model=self.name,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
            timings=batch_rewards_output.timings,
            uids=response_event.uids,
        )

        return reward_event
