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
            BatchRewardOutput: An object containing the computed rewards and timing details. Rewards are in the range [-3, 1].
        """
        all_chunks: list[list[str]] = response_event.stream_results_all_chunks
        all_timings: list[list[float]] = response_event.stream_results_all_chunks_timings
        timeout = response_event.timeout
        timing_outputs, rewards = [], []

        for chunks, timings in zip(all_chunks, all_timings):
            if not chunks:
                rewards.append(-PENALTY_FACTOR)
                timing_outputs.append(0)
                continue
                
            completion = "".join(chunks)
            if reference != completion:
                rewards.append(-PENALTY_FACTOR)
                timing_outputs.append(0)
                continue

            miner_reward = 0
            average_timing = []
            for chunk, timing in zip(chunks, timings):
                if chunk:
                    normalized_timing = min(1, max(0, ((timeout - timing) / timeout)))
                    average_timing.append(normalized_timing)
                    miner_reward += normalized_timing

            rewards.append(miner_reward / len(chunks))
            timing_outputs.append(np.array(average_timing).mean())

        output = BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(timing_outputs),
        )

        return output
