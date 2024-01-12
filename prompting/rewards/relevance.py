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

    @property
    def model_type(self) -> RewardModelTypeEnum:
        return RewardModelTypeEnum.WEIGHTED_REWARD

    def __init__(self, threshold=None, device="cuda", **kwargs):
        super().__init__()
        self.threshold = threshold
        self.model = AnglE.from_pretrained(
            "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
        )
        if device == "cuda":
            self.model = self.model.cuda()

    def reward(
        self, reference: str, completions: List[str]
    ) -> BatchRewardOutput:
        reference_embedding = self.model.encode(reference, to_numpy=False)
        completions_embeddings = self.model.encode(completions, to_numpy=False)

        rewards = cosine_similarity(
            reference_embedding, completions_embeddings, dim=1
        )

        output = BatchRewardOutput(
            rewards=rewards,
            extra_info={"threshold": self.threshold},
            timings=None,
        )

        return output
