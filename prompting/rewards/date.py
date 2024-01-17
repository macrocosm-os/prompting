import time
import torch
from typing import List
from prompting.rewards import BaseRewardModel, BatchRewardOutput, RewardModelTypeEnum


class DateRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return 'date'

    @property
    def model_type(self) -> RewardModelTypeEnum:
        return RewardModelTypeEnum.WEIGHTED_REWARD

    def __init__(self, **kwargs):
        super().__init__()

    def date_score(self, reference, completion):
        # TODO: cleanup code
        score = 1
        #Take the last 4 characters of the reference as the year
        year = reference[-4:]
        month = reference.split()[0].strip()
        month_num = str(time.strptime(month, "%B").tm_mon)
        day = reference.split()[1].strip(',')
        number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        not_in_month_day_year = set(str(month_num) + str(day) + str(year))
        numbers = [str(x) for x in number_list if str(x) not in not_in_month_day_year]
        # Create a list of the months
        month_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        months = [x for x in month_list if x not in month]
        
        if not year in completion:
            score -= 0.5
        if not (month_num in completion or month in completion):
            score -= 0.25
        if not day in completion:
            score -= 0.25
            
        if not score == 0:
            # Check if numbers are in completion
            for number in numbers:
                if str(number) in completion:
                    return 0.0
            # Check if months are in completion
            for month in months:
                if month in completion:
                    return 0.0
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
