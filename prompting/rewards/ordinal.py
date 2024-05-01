import time
import torch
from typing import List
from prompting.rewards import BaseRewardModel, BatchRewardOutput


class OrdinalRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "category_distance"

    def __init__(self, **kwargs):
        super().__init__()
        #TODO: Expand to allow for more than 3 classes (Must also adjust dataset/review.py)
        self.sentiments = [ 
            "positive",
            "neutral",
            "negative",
        ]

    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        classes = self.sentiments
        for completion in completions:
            t0 = time.time()
            completion = completion.lower()
            # Check if exactly one answer can be found in the completion
            if sum(option in completion for option in classes) == 1:
                answer = [option for option in classes if option in completion][0]
                reward = 1-abs(classes.index(reference) - classes.index(answer))/len(classes)
            else:
                reward = 0 
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
