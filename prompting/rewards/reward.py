import torch
import time
from typing import List
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
from mock import NetworkResponseEvent
from enum import Enum

class RewardModelTypeEnum(Enum):
    WEIGHTED_REWARD = 'weighted_reward'
    FILTER_REWARD = 'filter_reward'
    PENALTY = 'penalty'


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
            f'{self.model}_model_raw_rewards': self.rewards,
            f'{self.model}_model_timings':  [round(num, 5) for num in self.timings] if self.timings else None,
            f'{self.model}_model_batch_time': self.batch_time,
            f'{self.model}_model_extra_info': self.extra_info
        }


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
    def reward(self, reference:str, completions:List[str]) -> BatchRewardOutput:
        pass

    def apply(self, response_event: NetworkResponseEvent) -> RewardEvent:
        t0 = time.time()
        batch_rewards_output = self.reward(response_event.reference, response_event.completions)
        batch_rewards_time = time.time() - t0

        reward_event = RewardEvent(
            rewards = batch_rewards_output.rewards.cpu().tolist(),
            model_type = self.model_type,
            model = self.name,
            batch_time = batch_rewards_time,
            extra_info = batch_rewards_output.extra_info,
            timings = batch_rewards_output.timings,
            uids = response_event.uids
        )

        return reward_event

