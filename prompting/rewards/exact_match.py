import numpy as np
from typing import List
from prompting.rewards.reward import (
    BaseRewardModel,
    BatchRewardOutput,
)
from prompting.base.dendrite import DendriteResponseEvent


class ExactMatchRewardModel(BaseRewardModel):
    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Compute ROUGE scores given a completion and reference pair."""
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
