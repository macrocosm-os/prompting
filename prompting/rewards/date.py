import time
import torch
import re
import pandas as pd
import numpy as np
from typing import List
from prompting.rewards import BaseRewardModel, BatchRewardOutput, RewardModelTypeEnum


class DateRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return 'date'

    def __init__(self, **kwargs):
        super().__init__()


    
    def date_diff(self, ref_date, comp_date):
        """
        Calculates the absolute difference in days between two dates.
        """
        print(ref_date, comp_date)
        return abs(ref_date[0] - comp_date[0]).days + 365*abs(int(ref_date[1]) - int(comp_date[1]))
    
    def parse_dates_from_text(self, text: str):
        """
        Parses dates from a body of text, handling various formats, and returns pandas datetime objects.
        """

        date_patterns = [
            r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{3,4})\b",  # MM/DD/YYYY or DD/MM/YYYY
            r"\b(\d{3,4})[-/](\d{1,2})[-/](\d{1,2})\b",  # YYYY-MM-DD
            r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2})\b",   # MM/DD/YY or DD/MM/YY
            r"\b(\d{1,2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{3,4})\b",  # Month DD, YYYY
            r"\b(\d{3,4}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{1,2})\b",  # YYYY Month DD
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(,\s*)?(\d{3,4})\b",  # Month DD, YYYY
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Attempt to create a pandas datetime object
                    parsed_date = pd.to_datetime(match[0] + "/" + match[1] + "/" + "2000")
                    year = match[2]
                    return (parsed_date, year)
                except ValueError:
                    pass  # Ignore invalid date formats

        return dates
    
    def date_score(self, reference, completion):
        """Assign a score based on the difference between two dates using a negative exponential function."""
        score = 0
        if not completion:
            return score
        ref_date = self.parse_dates_from_text(reference)
        comp_date = self.parse_dates_from_text(completion)
        score = 1-np.exp(-self.date_diff(ref_date, comp_date)/5)
        return score

    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []

        for completion in completions:
            t0 = time.time()
            reward = self.date_score(reference, completion) 
            timings.append(time.time() - t0)
            rewards.append(reward)

        output = BatchRewardOutput(
            rewards = torch.FloatTensor(rewards),
            timings = torch.FloatTensor(timings),
            extra_info = {'type': 'date', },
        )
        return output
