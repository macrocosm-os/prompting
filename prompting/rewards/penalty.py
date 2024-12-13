import time

import numpy as np

from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from shared.dendrite import DendriteResponseEvent

NON_RESPONSE_PENALTY = 3


class PenaltyModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "penalty"

    def reward(self, reference: str, response_event: DendriteResponseEvent, **kwargs) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        completions: list[str] = response_event.completions
        t0 = time.perf_counter()

        for completion in completions:
            reward = -NON_RESPONSE_PENALTY if completion == "" else 0
            timings.append(time.perf_counter() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards=np.asarray(rewards),
            timings=np.asarray(timings),
        )
        return output
