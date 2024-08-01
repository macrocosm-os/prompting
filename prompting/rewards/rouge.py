import time
import torch
from typing import List
from rouge import Rouge
from prompting.rewards import (
    BaseRewardModel,
    BatchRewardOutput,
)
from prompting.dendrite import DendriteResponseEvent


class RougeRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "rouge"

    def __init__(self, ngram="rouge-l", metric="f", avg=False, device=None, **kwargs):
        super().__init__()
        self.ngram = ngram
        self.metric = metric
        self.avg = avg
        self.rouge = Rouge(**kwargs)

    def rouge_score(self, reference, completion):
        if not completion or not reference:
            return 0.0
        return self.rouge.get_scores(reference, completion, avg=self.avg)[0][self.ngram][self.metric]

    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Compute ROUGE scores given a completion and reference pair."""
        rewards = []
        timings = []
        completions: List[str] = response_event.completions

        for completion in completions:
            t0 = time.time()
            rewards.append(self.rouge_score(reference, completion))
            timings.append(time.time() - t0)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={
                "ngram": self.ngram,
                "metric": self.metric,
                "avg": self.avg,
            },
        )

        return output
