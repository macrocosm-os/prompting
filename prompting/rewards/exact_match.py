import numpy as np

from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from shared.dendrite import DendriteResponseEvent

PENALTY_FACTOR = 3


class ExactMatchRewardModel(BaseRewardModel):
    def reward(self, reference: str, response_event: DendriteResponseEvent, **kwargs) -> BatchRewardOutput:
        """Gives an exact reward of 1 if the response matches the reference, 0 otherwise"""
        rewards = []
        # completions: list[str] = response_event.completions
        all_chunks: list[list[str]] = response_event.stream_results_all_chunks
        all_timings: list[list[float]] = response_event.stream_results_all_chunks_timings

        for chunks, timings in zip(all_chunks, all_timings):
            completion = "".join(chunks)
            if reference != completion:
                rewards.append(-PENALTY_FACTOR)
                continue

            miner_reward = 0
            for chunk, timing in zip(chunks, timings):
                if chunk == "":
                    miner_reward = 0
                miner_reward += 1 - min(1, max(0, (timing / response_event.timeout)))
            rewards.append(miner_reward / len(chunks))

        output = BatchRewardOutput(
            rewards=np.array(rewards),
            timings=np.array(all_timings),
        )

        return output
