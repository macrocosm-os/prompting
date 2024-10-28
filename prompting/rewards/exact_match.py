import numpy as np
from prompting.rewards.reward import (
    BaseRewardModel,
    BatchRewardOutput,
)
from prompting.base.dendrite import DendriteResponseEvent


PENALTY_FACTOR = 3


class ExactMatchRewardModel(BaseRewardModel):
    def reward(self, reference: str, response_event: DendriteResponseEvent, **kwargs) -> BatchRewardOutput:
        """Gives an exact reward of 1 if the response matches the reference, 0 otherwise"""
        rewards = []
        completions: list[str] = response_event.completions
        timings = [0] * len(completions)

        for completion in completions:
            rewards.append(1 if reference == completion else -PENALTY_FACTOR)

        output = BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(timings),
        )

        return output
