import time
from typing import List

import numpy as np
from pydantic import ConfigDict
from rouge import Rouge

from shared.dendrite import DendriteResponseEvent
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput


class RougeRewardModel(BaseRewardModel):
    ngram: str = "rouge-l"  # TODO: Make proper literal
    metric: str = "f"  # TODO: Make proper literal
    avg: bool = False
    rouge: Rouge = Rouge()
    name: str = "rouge"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def rouge_score(self, reference, completion):
        if not completion or not reference:
            return 0.0
        return self.rouge.get_scores(reference, completion, avg=self.avg)[0][self.ngram][self.metric]

    def reward(self, reference: str, response_event: DendriteResponseEvent, **kwargs) -> BatchRewardOutput:
        """Compute ROUGE scores given a completion and reference pair."""
        rewards = []
        timings = []
        completions: List[str] = response_event.completions

        for completion in completions:
            t0 = time.time()
            rewards.append(self.rouge_score(reference, completion))
            timings.append(time.time() - t0)

        output = BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(timings),
        )

        return output
