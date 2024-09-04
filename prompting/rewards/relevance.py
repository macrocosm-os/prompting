import time
import numpy as np
from typing import List
from angle_emb import AnglE
from prompting.rewards.reward import (
    BaseRewardModel,
    BatchRewardOutput,
)
from prompting.base.dendrite import DendriteResponseEvent
from pydantic import model_validator, ConfigDict
from scipy import spatial
from prompting.settings import settings

MODEL = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy="cls", device=settings.NEURON_DEVICE)
if settings.NEURON_DEVICE.startswith("cuda"):
    # This line is necessary to pass the model to the device defined at its initialization
    MODEL = MODEL.cuda()


class RelevanceRewardModel(BaseRewardModel):
    threshold: float | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    embedding_model: AnglE | None = None

    @model_validator(mode="after")
    def init_model(self) -> "RelevanceRewardModel":
        self.embedding_model = MODEL
        return self

    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Calculates the cosine similarity between sentence embeddings of the reference and completions.
        We subtract a baseline score which is what an empty string would get (a failed completion). This is usually around 0.35
        We also clip the rewards between 0 and 1. The maximum effective score is around 0.65
        """
        reference_embedding = self.embedding_model.encode(reference, to_numpy=True)
        rewards = []
        timings = []
        completions: List[str] = response_event.completions
        # baseline is the cosine similarity between the reference and an empty string
        baseline = 1 - float(
            spatial.distance.cosine(
                reference_embedding.flatten(), self.embedding_model.encode("", to_numpy=True).flatten()
            )
        )

        for comp in completions:
            if len(comp) == 0:
                rewards.append(0)
                timings.append(0)
                continue
            t0 = time.time()
            emb = self.embedding_model.encode(comp, to_numpy=True)
            # Calculate cosine similarity between reference and completion embeddings, and subtract baseline
            score = spatial.distance.cosine(reference_embedding.flatten(), emb.flatten()) - baseline
            score = 1 - float(spatial.distance.cosine(reference_embedding.flatten(), emb.flatten() - baseline))

            rewards.append(score)
            timings.append(time.time() - t0)

        output = BatchRewardOutput(
            rewards=np.clip(np.array(rewards), 0, 1),
            timings=np.array(timings),
            threshold=self.threshold,
        )

        return output
