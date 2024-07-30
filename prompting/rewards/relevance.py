import time
import torch
from typing import List
from angle_emb import AnglE
from torch.nn.functional import cosine_similarity
from prompting.rewards.reward import (
    BaseRewardModel,
    BatchRewardOutput,
)
from prompting.dendrite import DendriteResponseEvent
from pydantic import model_validator, ConfigDict


class RelevanceRewardModel(BaseRewardModel):
    threshold: float | None = None
    model: AnglE | None = None
    pooling_strategy: str = "cls"
    device: str = "cuda"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def init_model(self) -> "RelevanceRewardModel":
        self.model = AnglE.from_pretrained(
            "WhereIsAI/UAE-Large-V1", pooling_strategy=self.pooling_strategy, device=self.device
        )
        if self.device.startswith("cuda"):
            # This line is necessary to pass the model to the device defined at its initialization
            self.model = self.model.cuda()
        return self

    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Calculates the cosine similarity between sentence embeddings of the reference and completions.
        We subtract a baseline score which is what an empty string would get (a failed completion). This is usually around 0.35
        We also clip the rewards between 0 and 1. The maximum effective score is around 0.65
        """
        reference_embedding = self.model.encode(reference, to_numpy=False)
        rewards = []
        timings = []
        completions: List[str] = response_event.completions
        # baseline is the cosine similarity between the reference and an empty string
        baseline = cosine_similarity(
            reference_embedding.reshape(1, -1),
            self.model.encode("", to_numpy=False).reshape(1, -1),
        )

        for comp in completions:
            t0 = time.time()

            emb = self.model.encode(comp, to_numpy=False)
            # Calculate cosine similarity between reference and completion embeddings, and subtract baseline
            score = cosine_similarity(reference_embedding.reshape(1, -1), emb.reshape(1, -1)) - baseline

            rewards.append(score)
            timings.append(time.time() - t0)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards).clip(min=0, max=1),
            timings=torch.FloatTensor(timings),
            extra_info={"threshold": self.threshold},
        )

        return output
