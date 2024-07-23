import time
import torch
import re
from typing import List
from prompting.rewards import BaseRewardModel, BatchRewardOutput
from prompting.dendrite import DendriteResponseEvent


class MultipleChoiceModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "multiple_choice"

    def __init__(self, **kwargs):
        super().__init__()
        self.choices = ("A", "B", "C", "D")

    def reward(
        self, reference: str, response_event: DendriteResponseEvent
    ) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        classes = self.choices
        completions: List[str] = response_event.completions

        for completion in completions:
            t0 = time.time()
            # step through the words in the completion and check if the reference is in the completion
            # remove any punctuation too.
            matches = [
                word
                for word in re.sub(r"\W", " ", completion).split()
                if word in classes
            ]

            # Take the last match as the answer
            if matches:
                reward = matches[-1] == reference
            else:
                reward = 0
            timings.append(time.time() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={
                "type": self.name,
            },
        )
        return output
