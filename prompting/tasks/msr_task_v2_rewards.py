from typing import ClassVar

import loguru as logger
import numpy as np

from prompting.datasets.msr_v2_dataset import MSRDiscriminatorDataset
from prompting.rewards.discriminator import DiscriminatorRewardModel
from prompting.rewards.reward import BaseRewardConfig, BaseRewardModel, BatchRewardOutput
from prompting.tasks.msr_task_v2 import (
    MultiStepReasoningRewardConfig,
    MultiStepReasoningTaskDiscriminator,
    MultiStepReasoningTaskGenerator,
)
from shared.dendrite import DendriteResponseEvent


class MSRv2GeneratorRewardModel(BaseRewardModel):
    async def reward(
        self,
        reference: str,
        response_event: DendriteResponseEvent,
        task: MultiStepReasoningTaskGenerator,
        task_queue: list,
        **kwargs,
    ) -> BatchRewardOutput:
        """Compute ROUGE scores given a completion and reference pair."""
        completions: list[str] = response_event.completions

        # There should only be one completion
        for completion, uid in zip(completions, response_event.uids):
            if completion and reference:
                MSRDiscriminatorDataset.add_entry(
                    miner_response=completion, validator_reference=reference, miner_uid=uid
                )
                task_queue.append(
                    MultiStepReasoningTaskDiscriminator(
                        dataset_entry=MSRDiscriminatorDataset.random(uid),
                        generator_task_id=task.task_id,  # Pass the Generator's task_id
                    )
                )
            else:
                logger.debug(f"Completion or reference is None: {completion} {reference}")
        return BatchRewardOutput(
            rewards=np.array([]),
            timings=np.array([]),
            uids=[],
        )


class MSRv2DiscriminatorRewardModel(BaseRewardModel):
    async def reward(
        self, reference: str, response_event: DendriteResponseEvent, task: MultiStepReasoningTaskDiscriminator, **kwargs
    ) -> BatchRewardOutput:
        """
        Compute ROUGE scores given a completion and reference pair.

        TODO: This needs to be implemented
        """
        # completions: list[str] = response_event.completions
        task.original_reference = reference
        return BatchRewardOutput(
            rewards=np.array([]),
            timings=np.array([]),
            uids=[],
        )


class MultiStepReasoningGeneratorRewardConfig(MultiStepReasoningRewardConfig):
    """The reward config for the generator task is the same as for the normal msr task"""

    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        MSRv2GeneratorRewardModel(weight=1),
    ]


class MultiStepReasoningDiscriminatorRewardConfig(BaseRewardConfig):
    reward_definitions: ClassVar[list[BaseRewardModel]] = [
        DiscriminatorRewardModel(weight=1),
    ]
