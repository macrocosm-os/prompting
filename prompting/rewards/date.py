import time
import torch
import re
import pandas as pd
import numpy as np
from typing import List
from prompting.rewards import BaseRewardModel, BatchRewardOutput
from prompting.dendrite import DendriteResponseEvent


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
        DATE_NOT_FOUND_CODE = 9999
        if not comp_date:
            return DATE_NOT_FOUND_CODE
        # Check if ref date is just a year
        if ref_date.isdigit():
            # Extract the last 3-4 digits from the completion date using a regex pattern that would detect 3 or 4 digit years
            comp_year = re.findall(r"\b\d{3,4}\b", comp_date)
            if comp_year:
                return abs(int(ref_date) - int(comp_year[0])) * 365
            else:
                return DATE_NOT_FOUND_CODE
        # If the reference date is not only a year, take the difference between the two dates
        try:
            ref_date = pd.to_datetime(ref_date)
            comp_date = pd.to_datetime(comp_date)
            return abs((ref_date - comp_date).days)
        except Exception as _:
            if ref_date == comp_date:
                return 0
            else:
                return DATE_NOT_FOUND_CODE

    def parse_dates_from_text(self, text: str) -> tuple:
        # Regular expression to find dates in various formats
        date_pattern = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember))\s+\d{4}\b|\b\d{4}\b"

        # Compile the regex pattern
        date_regex = re.compile(date_pattern)

        # Split text into sentences
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

        # Initialize dictionary to store results

        # Iterate through sentences and find dates
        for sentence in sentences:
            # Find all dates in the sentence
            dates = date_regex.findall(sentence)
            # If dates are found, add them to the result dictionary with the corresponding sentence
            if dates:
                return dates[0]
        return None

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

    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair.

        Args:
            reference (str): The reference date.
            completions (List[str]): A list of completions.

        Returns:
            BatchRewardOutput: A BatchRewardOutput object containing the rewards and timings.
        """
        completions: List[str] = response_event.completions
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
