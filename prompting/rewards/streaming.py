import time
import torch
from prompting.dendrite import DendriteResponseEvent
from prompting.rewards import (
    BaseRewardModel,
    BatchRewardOutput,
)

## TODO: Create unit tests
class StreamingRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "streaming"

    def __init__(self, max_tokens_per_chunk:int, **kwargs):
        super().__init__()
        self.max_tokens_per_chunk = max_tokens_per_chunk


    def reward(self, _: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""       
        rewards = []
        timings = []
        penalty_per_exceeding_chunk = 0.25
                
        # Iterate through each chunk of response tokens
        for response_tokens_per_chunks in response_event.stream_results_all_tokens_per_chunk:            
            start_time = time.time()
             
            # Calculate the accumulated penalty for the current chunk
            accumulated_penalty = sum(
                penalty_per_exceeding_chunk if tokens_per_chunk > self.max_tokens_per_chunk else 0 
                for tokens_per_chunk in response_tokens_per_chunks
            )
            
            # Record the timing for this computation
            timings.append(time.time() - start_time)
            
            # Calculate the reward and ensure it does not go above 1
            rewards.append(min(accumulated_penalty, 1))
        
        # Create the output object with rewards, timings, and extra information
        output = BatchRewardOutput(
            rewards=torch.FloatTensor(rewards),
            timings=torch.FloatTensor(timings),
            extra_info={"type": "streaming"}
        )
        return output

