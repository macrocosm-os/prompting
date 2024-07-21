import time
import torch
from typing import List
from prompting.rewards import BaseRewardModel, BatchRewardOutput
from prompting.dendrite import DendriteResponseEvent


class OrdinalRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "ordinal"

    def __init__(self, **kwargs):
        super().__init__()
        # TODO: Expand to allow for more than 3 classes (Must also adjust dataset/review.py)
        self.sentiments: list[str] = [
            "positive",
            "neutral",
            "negative",
        ]

    def reward(
        self, reference: str, response_event: DendriteResponseEvent
    ) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        classes = self.sentiments
        completions: list[str] = response_event.completions

        for completion in completions:
            t0 = time.time()
            completion = completion.lower()
            # Check if exactly one answer can be found in the completion
            if sum(option in completion for option in classes) == 1:
                answer = [option for option in classes if option in completion][0]
                reward = 1 - abs(classes.index(reference) - classes.index(answer)) / (
                    len(classes) - 1
                )
            else:
                reward = 0
            timings.append(time.time() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={
                "type": "ordinal",
            },
        )
        return output
