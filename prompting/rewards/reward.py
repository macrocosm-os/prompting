import torch
import time
import bittensor as bt
from typing import List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class RewardModelTypeEnum(Enum):
    WEIGHTED_REWARD = "weighted_reward"
    FILTER_REWARD = "filter_reward"
    PENALTY = "penalty"


@dataclass
class RewardEvent:
    """Contains rewards for all the responses in a batch"""
    rewards: torch.FloatTensor
    timings: torch.FloatTensor
    model_name: str
    model_type: RewardModelTypeEnum
    batch_time: float
    extra_info: dict

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> dict:
        return {
            f"{self.model_name}_raw_rewards": self.rewards,
            f"{self.model_name}_timings": self.timings,
            f"{self.model_name}_batch_time": self.batch_time,
            f"{self.model_name}_extra_info": self.extra_info,
        }


class RewardResult:
    def __init__(self, reward_pipeline, task, response_event, device):

        self.reward_pipeline = reward_pipeline
        self.response_event = response_event
        self.device = device
        self.task = task
        self.reward_events = self.reward_responses()
        self.rewards = self.total_reward()

    def reward_responses(self) -> List[RewardEvent]:
        """Calculates the rewards for the responses given the task and returns a RewardEvent for each reward model"""

        reward_events = []

        for reward_info in self.task.reward_definition:

            # Select the reward model from preloaded reward model pipeline
            model = self.reward_pipeline.reward_models.get(reward_info["name"])
            if not model:
                raise ValueError(
                    f"Reward model {reward_info['name']} not supported. Please choose from {self.reward_pipeline.reward_models.keys()}"
                )

            # Compute the rewards for the responses given the prompt
            reward_event = model.apply(self.response_event)
            reward_events.append(reward_event.to(self.device))

        return reward_events

    def total_reward(self) -> torch.FloatTensor:
        """Combines the rewards from all the reward models into a single reward tensor"""

        # TODO: How would using the Agent as a reward model fit into this flow?
        # Compute the rewards for the responses given the prompt
        rewards = torch.zeros_like(self.response_event.uids, dtype=torch.float16)

        for reward_type in RewardModelTypeEnum:

            for reward_info in self.task.reward_definition:
                bt.logging.info(f"reward_info: {reward_info}")

                for reward_event in filter(
                    lambda x: x.model_name == reward_info["name"] and reward_event.model_type == reward_type,
                    self.rewards_events
                    ):

                    # Gets appropriate reward event for the reward model defined in the task
                    if reward_type == RewardModelTypeEnum.WEIGHTED_REWARD:

                        rewards += reward_info.get("weight", 1) * reward_event.rewards

                    else:
                        raise ValueError(
                            f"Reward model type {reward_event.model_type} not supported."
                        )

        return rewards.to(self.device)


@dataclass
class BatchRewardOutput:
    rewards: torch.FloatTensor
    timings: torch.FloatTensor
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
            model_name=self.name,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
            timings=batch_rewards_output.timings,
            uids=response_event.uids,
        )

        return reward_event
