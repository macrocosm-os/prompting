from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel
from shared.dendrite import DendriteResponseEvent
from prompting.rewards.reward import BatchRewardOutput
from prompting.datasets.msr_v2_dataset import MSRDiscriminatorDataset
import time
import numpy as np
from prompting.tasks.msr_task_v2 import MultiStepReasoningTaskGenerator, MultiStepReasoningTaskDiscriminator
class MSRv2GeneratorRewardModel(BaseRewardModel):
    async def reward(self, reference: str, response_event: DendriteResponseEvent, **kwargs) -> BatchRewardOutput:
        """Compute ROUGE scores given a completion and reference pair."""
        completions: list[str] = response_event.completions

        # There should only be one completion
        for completion, uid in zip(completions, response_event.uids):
            MSRDiscriminatorDataset.add_entry(miner_response=completion, validator_reference=reference, miner_uid=uid)
        return BatchRewardOutput(
            rewards=np.array([]),
            timings=np.array([]),
        )
            
class MSRv2DiscriminatorRewardModel(BaseRewardModel):
    async def reward(self, reference: str, response_event: DendriteResponseEvent, task: MultiStepReasoningTaskDiscriminator, **kwargs) -> BatchRewardOutput:
        """Compute ROUGE scores given a completion and reference pair."""
        completions: list[str] = response_event.completions
        task.original_reference
