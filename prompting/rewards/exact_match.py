import numpy as np
from typing import List
from prompting.rewards.reward import (
    BaseRewardModel,
    BatchRewardOutput,
)
from prompting.base.dendrite import DendriteResponseEvent


class ExactMatchRewardModel(BaseRewardModel):
    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Gives an exact reward of 1 if the response matches the reference, 0 otherwise"""
        rewards = []
        timings = []
        completions: List[str] = response_event.completions

        for completion in completions:
            rewards.append(reference == completion)

        output = BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(timings),
        )

        return output
