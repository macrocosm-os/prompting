import re
import time

import numpy as np

from prompting.base.dendrite import DendriteResponseEvent
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput


class MultiChoiceRewardModel(BaseRewardModel):
    choices: tuple[str, str, str, str] = ("A", "B", "C", "D")

    @property
    def name(self) -> str:
        return "multiple_choice"

    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        classes = self.choices
        completions: list[str] = response_event.completions

        for completion in completions:
            t0 = time.perf_counter()
            # Step through the words in the completion and check if the reference is in the completion.
            # Remove any punctuation too.
            matches = [
                word
                for word in re.sub(r"\W", " ", completion).split()
                if word in classes
            ]

            # Take the last match as the answer
            if matches:
                reward = matches[-1] == reference
            else:
                reward = 0
            timings.append(time.perf_counter() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards=np.asarray(rewards),
            timings=np.asarray(timings),
            # extra_info={
            #     "type": self.name,
            # },
        )
        return output
