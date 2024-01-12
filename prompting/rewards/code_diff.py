import difflib
import torch
from typing import List
from prompting.rewards import (
    BaseRewardModel,
    BatchRewardOutput,
    RewardModelTypeEnum,
)
import time


class DiffRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "diff"

    @property
    def model_type(self) -> RewardModelTypeEnum:
        return RewardModelTypeEnum.WEIGHTED_REWARD

    def __init__(self, lines=False, threshold=None, **kwargs):
        super().__init__()
        self.lines = lines
        self.threshold = threshold

    def unified_diff(self, reference, completion):
        return len(
            difflib.unified_diff(
                reference.splitlines(), completion.splitlines()
            )
        )

    def seq_match(self, reference, completion):
        return difflib.SequenceMatcher(None, reference, completion).ratio()

    def reward(
        self, reference: str, completions: List[str]
    ) -> BatchRewardOutput:
        """Get the score between two strings.
        lines: If True, return a unified diff. If False, return a ratio.
        """

        rewards = []
        timings = []

        if self.lines:
            for completion in completions:
                t0 = time.time()
                rewards.append(self.unified_diff(reference, completion))
                timings.append(time.time() - t0)
        else:
            for completion in completions:
                t0 = time.time()
                rewards.append(self.seq_match(reference, completion))
                timings.append(time.time() - t0)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            extra_info={"threshold": self.threshold, "lines": self.lines},
            timings=timings,
        )

        return output
