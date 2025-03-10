from typing import TYPE_CHECKING

import numpy as np

from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from shared.dendrite import DendriteResponseEvent

if TYPE_CHECKING:
    from prompting.tasks.msr_task_v2 import MultiStepReasoningTaskDiscriminator


class DiscriminatorRewardModel(BaseRewardModel):
    """
    This reward model is used to reward the discriminator task by comparing the reference and the response.
    """

    async def reward(
        self,
        reference: str,
        response_event: DendriteResponseEvent,
        task: "MultiStepReasoningTaskDiscriminator",
        **kwargs,
    ) -> BatchRewardOutput:
        completions: list[str] = response_event.completions

        # Get miner_uid from either original_miner_uid or dataset_entry
        miner_uid = task.original_miner_uid  # if task.original_miner_uid is not None else task.dataset_entry.miner_uid
        rewards: list[float] = []

        for completion in completions:
            rewards.append(1 / len(completions) if completion == reference else 0)

        generator_reward = 1 - np.sum(rewards)
        # Convert to list and use the miner_uid we retrieved
        uids = [float(miner_uid)] + list(response_event.uids)
        rewards = [generator_reward] + rewards

        return BatchRewardOutput(rewards=np.array(rewards), timings=np.array([0] * len(rewards)), uids=uids)
