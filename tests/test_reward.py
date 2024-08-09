import pytest
import torch
from unittest.mock import MagicMock
from prompting.rewards import StreamingRewardModel


@pytest.mark.parametrize(
    "all_tokens_per_chunk, expected_rewards",
    [
        ([[1, 1, 1, 1, 1]], [0]),  # No penalties
        ([[2, 1, 1, 1, 1]], [0.25]),  # First chunk exceeds
        ([[2, 2, 1, 1, 1]], [0.5]),  # Two chunks exceed
        ([[2, 2, 2, 1, 1]], [0.75]),  # Three chunks exceed
        ([[2, 2, 2, 2, 1]], [1]),  # Four chunks exceed
        ([[2, 2, 2, 2, 2, 2]], [1]),  # Sum of chunks > 1, clipped at 1
    ],
)
def test_streaming_reward_model(all_tokens_per_chunk, expected_rewards):
    max_tokens_per_chunk = 1
    response_event = MagicMock()
    response_event.stream_results_all_tokens_per_chunk = all_tokens_per_chunk

    model = StreamingRewardModel(max_tokens_per_chunk)

    output = model.reward("", response_event)

    assert torch.allclose(
        output.rewards, torch.tensor(expected_rewards, dtype=torch.float)
    ), f"Expected rewards {expected_rewards} but got {output.rewards}"


if __name__ == "__main__":
    pytest.main()
