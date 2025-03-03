from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from shared.dendrite import DendriteResponseEvent
from prompting.tasks.base_task import BaseTask
from prompting.tasks.msr_task_v2 import MultiStepReasoningTaskDiscriminator
import numpy as np

class DiscriminatorRewardModel(BaseRewardModel):
    async def reward(self, reference: str, response_event: DendriteResponseEvent, task: MultiStepReasoningTaskDiscriminator , **kwargs) -> BatchRewardOutput:
        completions: list[str] = response_event.completions
        task.original_miner_uid
        rewards: list[float] = []

        for completion in completions:
            rewards.append(1/len(completions) if completion == reference else 0)
        
        generator_reward = 1 - np.sum(rewards)
        uids = [task.original_miner_uid] + response_event.uids
        rewards = [generator_reward] + rewards
        return BatchRewardOutput(rewards=np.array(rewards), timings=np.array([0]*len(rewards)), uids=uids)
