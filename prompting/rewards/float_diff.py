import time
import torch
from typing import List
from rewards import BaseRewardModel, BatchRewardOutput, RewardModelTypeEnum


class FloatDiffModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return 'float_diff'

    @property
    def model_type(self) -> RewardModelTypeEnum:
        return RewardModelTypeEnum.WEIGHTED_REWARD

    def __init__(self, **kwargs):
        super().__init__()
        # self.ngram = ngram
        # self.metric = metric
        # self.avg = avg
        # TODO: Add init args to Rouge if required
        # self.rouge = Rouge()

    def math_score(self, reference, completion):
        # Extract all the digits and . from the completion
        completion_digits = [char for char in completion if char.isdigit() or char == '.']
        # Convert the list of digits to a string
        completion_digits = ''.join(completion_digits)
        try:
            # Convert the string to a float
            completion_digits = float(completion_digits)
            # Convert the reference to a float
            reference = float(reference)
            if completion_digits == reference:
                return 1.0
            # Compute the difference
            diff = abs(reference - completion_digits)/(reference + 1e-6)
            # Make sure the difference is between 0 and 1
            diff = min(abs(diff), 1)
            return 1.0 - diff
        except ValueError:
            return 0.0

    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []

        for completion in completions:
            t0 = time.time()
            reward = self.math_score(reference, completion) 
            timings.append(time.time() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards = torch.FloatTensor(rewards),
            timings = timings,
            extra_info = {'type': 'math', },
        )
        return output
