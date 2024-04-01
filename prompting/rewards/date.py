import time
import sentry_sdk
import torch
import re
import pandas as pd
import numpy as np
from typing import List
from prompting.rewards import BaseRewardModel, BatchRewardOutput, RewardModelTypeEnum
import bittensor as bt


class DateRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "date"

    def __init__(self, **kwargs):
        super().__init__()

    def date_diff(self, ref_date: tuple, comp_date: tuple) -> int:
        """
        Calculates the absolute difference in days between two dates.
        """
        try:
            return abs(ref_date[0] - comp_date[0]).days + 365 * abs(
                int(ref_date[1]) - int(comp_date[1])
            )
        except Exception as e:
            sentry_sdk.capture_exception()
            return 500

    def parse_dates_from_text(self, text: str) -> tuple:
        """
        Parses dates from a body of text, handling various formats, and returns pandas datetime objects.

        Args:
            text (str): The text to parse.

        Returns:
            tuple: A tuple containing a datemtime object with they year set at 2000 and the actual year.
        """

        date_patterns = [
            r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{3,4})\b",  # MM/DD/YYYY or DD/MM/YYYY
            r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2})\b",  # MM/DD/YY or DD/MM/YY
            r"\b(\d{1,2}) (January|February|March|April|May|June|July|August|September|October|November|December) (\d{3,4})\b",  # DD Month, YYYY
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(,\s*)?(\d{3,4})\b",  # Month DD, YYYY
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Attempt to create a datetime object with year 2000 (datetime objects cannot take dates more than 200 years in the past)
                    parsed_date = pd.to_datetime(
                        match[0] + "/" + match[1] + "/" + "2000"
                    )
                    year = match[-1]
                    # Check if the year is a number
                    if year.isdigit():
                        # If the year is a digit, return the parsed date and the year in a tuple
                        return (parsed_date, year)
                    else:
                        raise ValueError
                except ValueError:
                    sentry_sdk.capture_exception()
                    pass

        return

    def date_score(self, reference: str, completion: str) -> float:
        """Assign a score based on the difference between two dates using a negative exponential function.

        Args:
            reference (str): The reference date.
            completion (str): The completion date.

        Returns:
            float: The score."""
        score = 0
        if not completion:
            return score
        ref_date = self.parse_dates_from_text(reference)
        comp_date = self.parse_dates_from_text(completion)
        score = np.exp(-(self.date_diff(ref_date, comp_date) ** 2 / 1000))
        # Clip any very small scores
        if score < 0.001:
            score = 0
        return score

    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair.

        Args:
            reference (str): The reference date.
            completions (List[str]): A list of completions.

        Returns:
            BatchRewardOutput: A BatchRewardOutput object containing the rewards and timings.
        """
        rewards = []
        timings = []

        for completion in completions:
            t0 = time.time()
            reward = self.date_score(reference, completion)
            timings.append(time.time() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={
                "type": "date",
            },
        )
        return output
