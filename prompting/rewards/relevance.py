import time
import torch
from typing import List
from angle_emb import AnglE
from torch.nn.functional import cosine_similarity
from prompting.rewards import (
    BaseRewardModel,
    BatchRewardOutput,
    RewardModelTypeEnum,
)


class RelevanceRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "relevance"

    def __init__(self, threshold=None, device=None, pooling_strategy="cls"):
        super().__init__()
        self.threshold = threshold
        self.model = AnglE.from_pretrained(
            "WhereIsAI/UAE-Large-V1", pooling_strategy=pooling_strategy
        )
        if device == "cuda":
            self.model = self.model.cuda()

    def reward(
        self, reference: str, completions: List[str]
    ) -> BatchRewardOutput:
        reference_embedding = self.model.encode(reference, to_numpy=False)
        rewards = []
        timings = []

        for comp in completions:
            t0 = time.time()
            score = 0
            if comp:
                emb = self.model.encode(completions, to_numpy=False)
                score = cosine_similarity(reference_embedding.reshape(1, -1), emb.reshape(1, -1))

            rewards.append(score)
            timings.append(time.time() - t0)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={"threshold": self.threshold},
        )

        return output
