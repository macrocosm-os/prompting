import time
import torch
from typing import List
from prompting.rewards import (
    BaseRewardModel,
    BatchRewardOutput,
)
from prompting.dendrite import DendriteResponseEvent


class StreamingRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "streaming"

    def __init__(self, max_tokens_per_chunk:int, **kwargs):
        super().__init__()
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def reward(self, _: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        pass
