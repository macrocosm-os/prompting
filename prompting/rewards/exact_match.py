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
        completions: list[str] = response_event.completions
        timeout = response_event.timeout
        timing_outputs, rewards = [], []

        for chunks, timings, completion in zip(all_chunks, all_timings, completions):
            if chunks == []:
                rewards.append(-PENALTY_FACTOR)
                timing_outputs.append(0)
                continue

            # If the completion is a prefix of the reference, give a less severe penalty
            if len(completion) < len(reference) and reference.startswith(completion):
                rewards.append(-PENALTY_FACTOR * 0.33)
                timing_outputs.append(0)
                continue

            if reference != completion:
                rewards.append(-PENALTY_FACTOR)
                timing_outputs.append(0)
                continue

            # add way of calculating average time per token
            valid_chunks = []
            for chunk, timing in zip(chunks, timings):
                if chunk != []:
                    normalized_timing = min(1, max(0, ((timeout - timing) / timeout)))
                    valid_chunks.append(normalized_timing)                    
            if valid_chunks:
                final_score = np.mean(valid_chunks)  # This will be between 0 and 1.
            else:
                final_score = -PENALTY_FACTOR
            rewards.append(float(final_score))
            timing_outputs.append(np.array(valid_chunks).mean())

        output = BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(timing_outputs),
        )

        logger.debug("=== Reference ===")
        logger.debug(reference)
        logger.debug("=== Completions ===")
        logger.debug(completions)
        logger.debug("=== Rewards ===")
        logger.debug(rewards)
        logger.debug("=== Timings ===")
        logger.debug(timing_outputs)

        return output
