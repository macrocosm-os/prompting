import time
import sentry_sdk
import torch
from typing import List
from sympy.parsing.sympy_parser import parse_expr
from prompting.rewards import BaseRewardModel, BatchRewardOutput, RewardModelTypeEnum


class FloatDiffModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "float_diff"

    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def extract_number(text: str) -> float:
        """Extract a number from a string."""
        # loop over all words reversed and try to cast as a float, break when you find the first one
        words = text.split()
        for word in reversed(words):
            cleaned = word.strip(".").replace(",", "")
            try:
                return float(parse_expr(cleaned).evalf())
            except Exception:
                sentry_sdk.capture_exception()
                # fall back to simpler parsing if required
                try:
                    return float(cleaned)
                except Exception:
                    sentry_sdk.capture_exception()
                    continue

    @staticmethod
    def math_score(reference: str, completion: str) -> float:
        """Compute a score based on the difference between a reference and a completion."""
        # Convert the strings to a float
        reference = float(reference)
        pred = FloatDiffModel.extract_number(completion)
        if pred is None:
            return 0.0

        try:
            if pred == reference:
                return 1.0
            # Compute the difference
            diff = (reference - pred) / (reference + 1e-10)
            # Make sure the difference is between 0 and 1
            diff = min(abs(diff), 1)
            # Clip any very small scores
            if diff > 0.999:
                diff = 1.0
            return 1.0 - diff
        except Exception:
            sentry_sdk.capture_exception()
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
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={
                "type": "math",
            },
        )
        return output
