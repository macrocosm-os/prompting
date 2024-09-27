import re
import time
import json

import numpy as np

from prompting.base.dendrite import DendriteResponseEvent
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput


class MultiChoiceRewardModel(BaseRewardModel):
    choices: tuple[str, str, str, str] = ("A", "B", "C", "D")

    @property
    def name(self) -> str:
        return "multiple_choice"

    @staticmethod
    def safe_load_json(json_string):
        """Load a JSON string safely."""
        # Strip leading and trailing spaces from the entire string
        cleaned_json_string = json_string.strip()

        # Remove trailing commas from JSON objects and arrays
        cleaned_json_string = re.sub(r",(\s*[}\]])", r"\1", cleaned_json_string)

        # Remove newline characters inside key-value strings
        cleaned_json_string = re.sub(r'"\s*\n\s*"', r'""', cleaned_json_string)

        try:
            # Attempt to load the cleaned string as JSON
            return json.loads(cleaned_json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")

    def process_predictions(self, predictions: str) -> str:
        """Process the predictions from the miners and validate the data.."""

        probs = predictions.values()
        # Check that values are numeric
        if not all(isinstance(value, (int, float)) for value in probs):
            raise ValueError("Values must be numeric")

        # Extract predictions for allowed choices
        valid_choices = {p: v for p, v in probs.items() if p.upper() in self.choices}

        # Check that values sum to 1
        if not np.isclose(sum(valid_choices.values()), 1.0):
            raise ValueError("Values must sum to 1")

        return valid_choices

    def _logit_reward(self, reference: str, completion: str) -> float:
        """Compute difference scores given a completion and reference pair."""
        try:
            loaded_json = self.safe_load_json(completion)
            valid_choices = self.process_predictions(loaded_json)
            return valid_choices.get(reference.upper(), 0)

        except ValueError as e:
            return 0

    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        completions: list[str] = response_event.completions

        for completion in completions:
            t0 = time.perf_counter()
            # Convert completion to a dictionary and extract the logit probability for the reference cho
            reward = self._logit_reward(reference, completion)
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
