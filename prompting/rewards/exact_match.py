import numpy as np
from loguru import logger

from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from shared.dendrite import DendriteResponseEvent

PENALTY_FACTOR = 3


class ExactMatchRewardModel(BaseRewardModel):
    def reward(self, reference: str, response_event: DendriteResponseEvent, **kwargs) -> BatchRewardOutput:
        """
        Calculates rewards based on the exact match of the response with the reference string.

        - If the response matches the reference:
            - Rewards are calculated based on timing metrics for each chunk.
            - Slower chunks receive reduced rewards
        - If the response does not match the reference, a penalty is applied.

        Parameters:
            reference (str): The expected response string.
            response_event (DendriteResponseEvent): Contains completions, timings, and other details.

        Returns:
            BatchRewardOutput: An object containing the computed rewards and timing details.
        """
        rewards = []
        all_chunks: list[list[str]] = response_event.stream_results_all_chunks
        all_timings: list[list[float]] = response_event.stream_results_all_chunks_timing
        timeout = response_event.timeouts

        debug_output = {
            # "reference": reference,
            "all_chunks": all_chunks,
            "all_timings": all_timings,
            "timeout": timeout,
        }

        logger.error(str(debug_output) + "£££££")

        for chunks, timings in zip(all_chunks, all_timings):
            completion = "".join(chunks)
            if reference != completion:
                # Apply penalty if there is no exact match
                rewards.append(-PENALTY_FACTOR)
                continue

            miner_reward = 0
            for chunk, timing in zip(chunks, timings):
                if chunk:
                    normalized_timing = min(1, max(0, (timing / timeout)))
                    miner_reward += normalized_timing

            rewards.append(miner_reward / len(chunks))

        output = BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(all_timings),
        )

        return output
