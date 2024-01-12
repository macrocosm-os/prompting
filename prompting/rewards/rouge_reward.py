import time
import torch
import bittensor as bt
from typing import List
from rouge import Rouge
from prompting.rewards import (
    BaseRewardModel,
    BatchRewardOutput,
    RewardModelTypeEnum,
)


class RougeRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "rouge"

    @property
    def model_type(self) -> RewardModelTypeEnum:
        return RewardModelTypeEnum.WEIGHTED_REWARD

    def __init__(self, ngram="rouge-l", metric="f", avg=False, **kwargs):
        super().__init__()
        self.ngram = ngram
        self.metric = metric
        self.avg = avg
        self.rouge = Rouge(**kwargs)

    def rouge_score(self, reference, completion):
        if not completion or not reference:
            return 0.0
        return self.rouge.get_scores(reference, completion, avg=self.avg)[0][
            self.ngram
        ][self.metric]

    def reward(
        self, reference: str, completions: List[str]
    ) -> BatchRewardOutput:
        """Compute ROUGE scores given a completion and reference pair."""
        rewards = []
        timings = []

        for completion in completions:
            t0 = time.time()
            self.rouge_score(reference, completion)
            timings.append(time.time() - t0)
            rewards.append(self.rouge_score(reference, completion))

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=timings,
            extra_info={
                "ngram": self.ngram,
                "metric": self.metric,
                "avg": self.avg,
            },
        )

        return output
