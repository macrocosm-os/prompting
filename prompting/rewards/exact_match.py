import numpy as np
from prompting.rewards.reward import (
    BaseRewardModel,
    BatchRewardOutput,
)
from prompting.base.dendrite import DendriteResponseEvent


class ExactMatchRewardModel(BaseRewardModel):
    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Gives an exact reward of 1 if the response matches the reference, 0 otherwise"""
        rewards = []
        completions: list[str] = response_event.completions
        timings = [0] * len(completions)

        for completion in completions:
            rewards.append(reference == completion)

        output = BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(timings),
        )

        return output
