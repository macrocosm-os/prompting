import time

import numpy as np

from prompting.base.dendrite import DendriteResponseEvent
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput

PENALTY_FACTOR = 0


class PenaltyModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "penalty"

    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        completions: list[str] = response_event.completions
        t0 = time.perf_counter()

        for completion in completions:
            reward = -PENALTY_FACTOR if completion == "" else 0
            timings.append(time.perf_counter() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards=np.asarray(rewards),
            timings=np.asarray(timings),
        )
        return output
