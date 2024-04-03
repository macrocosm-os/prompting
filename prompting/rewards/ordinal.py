import time
import torch
from typing import List
from prompting.rewards import BaseRewardModel, BatchRewardOutput


class CategoricalDistance(BaseRewardModel):
    @property
    def name(self) -> str:
        return "category_distance"

    def __init__(self, **kwargs):
        super().__init__()

    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        classes = self.sentiments

        for completion in completions:
            t0 = time.time()
            reward = abs(classes.index(reference) - classes.index(completion))
            timings.append(time.time() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={
                "type": "math",
            },
        )
        return output
