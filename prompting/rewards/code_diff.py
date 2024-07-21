import difflib
import torch
from typing import List
from prompting.rewards import (
    BaseRewardModel,
    BatchRewardOutput,
    RewardModelTypeEnum,
)
from prompting.dendrite import DendriteResponseEvent
import time


class DiffRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "diff"

    def __init__(self, lines=False, threshold=None, **kwargs):
        super().__init__()
        self.lines = lines
        self.threshold = threshold

    def unified_diff(self, reference, completion):
        return len(
            difflib.unified_diff(reference.splitlines(), completion.splitlines())
        )

    def seq_match(self, reference, completion):
        return difflib.SequenceMatcher(None, reference, completion).ratio()

    def reward(
        self, reference: str, response_event: DendriteResponseEvent
    ) -> BatchRewardOutput:
        """Get the score between two strings.
        lines: If True, return a unified diff. If False, return a ratio.
        """
        rewards = []
        timings = []
        completions: List[str] = response_event.completions

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
            timings=torch.FloatTensor(timings),
            extra_info={"threshold": self.threshold, "lines": self.lines},
        )

        return output
