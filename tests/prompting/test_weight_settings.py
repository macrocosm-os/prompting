# ruff: noqa: E402
import asyncio

import numpy as np

from shared import settings

settings.settings = settings.Settings(mode="mock")
raw_rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
from unittest.mock import MagicMock, patch

from prompting.weight_setting.weight_setter import WeightSetter, apply_reward_func


def test_apply_reward_func():
    raw_rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # test result is even returned
    result = apply_reward_func(raw_rewards)
    assert result is not None, "Result was None"

    # Test with p = 0.5 (no change)
    result = apply_reward_func(raw_rewards, p=0.5)
    assert np.allclose(
        result, raw_rewards / np.sum(raw_rewards), atol=1e-6
    ), "Result should be unchanged from raw rewards"

    # Test with p = 0 (more linear)
    result = apply_reward_func(raw_rewards, p=0)
    assert np.isclose(np.std(result), 0, atol=1e-6), "All rewards should be equal"

    # Test with p = 1 (more exponential)
    result = apply_reward_func(raw_rewards, p=1)
    assert result[-1] > 0.9, "Top miner should take vast majority of reward"

    # Test with negative values
    raw_rewards = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    result = apply_reward_func(raw_rewards, p=0.5)
    assert result[0] < 0, "Negative reward should remain negative"


def test_run_step_with_reward_events():
    with (
        patch("shared.uids.get_uids") as mock_get_uids,
        patch("prompting.weight_setting.weight_setter.TaskRegistry") as MockTaskRegistry,
        patch("prompting.weight_setting.weight_setter.mutable_globals") as mock_mutable_globals,
        patch("prompting.weight_setting.weight_setter.set_weights") as mock_set_weights,
        patch("prompting.weight_setting.weight_setter.logger") as mock_logger,
    ):

        class MockTask:
            pass

        class TaskConfig:
            def __init__(self, name, probability):
                self.name = name
                self.probability = probability
                self.task = MockTask

        class WeightedRewardEvent:
            def __init__(self, task, uids, rewards, weight):
                self.task = task
                self.uids = uids
                self.rewards = rewards
                self.weight = weight

        mock_uids = [1, 2, 3, 4, 5]
        mock_get_uids.return_value = mock_uids

        # Set up the mock TaskRegistry
        mock_task_registry = MockTaskRegistry
        mock_task_registry.task_configs = [
            TaskConfig(name="Task1", probability=0.5),
        ]
        mock_task_registry.get_task_config = MagicMock(return_value=mock_task_registry.task_configs[0])

        # Set up the mock mutable_globals
        mock_mutable_globals.reward_events = [
            [
                WeightedRewardEvent(
                    task=mock_task_registry.task_configs[0], uids=mock_uids, rewards=[1.0, 2.0, 3.0, 4.0, 5.0], weight=1
                ),
            ],
            [
                WeightedRewardEvent(
                    task=mock_task_registry.task_configs[0], uids=mock_uids, rewards=[5.0, 4.0, 3.0, 2.0, 1.0], weight=1
                ),
            ],
        ]

        weight_setter = WeightSetter()
        output = asyncio.run(weight_setter.run_step())

        print(output)
        mock_set_weights.assert_called_once()
        call_args = mock_set_weights.call_args[0]
        assert len([c for c in call_args[0] if c > 0]) == len(mock_uids)
        assert np.isclose(np.sum(call_args[0]), 1, atol=1e-6)

        # Check that the warning about empty reward events is not logged
        mock_logger.warning.assert_not_called()


# def test_run_step_without_reward_events(weight_setter):
#     with (
#         patch("prompting.weight_setter.get_uids") as mock_get_uids,
#         patch("prompting.weight_setter.TaskRegistry.task_configs", new_callable=property) as mock_task_configs,
#         patch("prompting.weight_setter.mutable_globals.reward_events") as mock_reward_events,
#         patch("prompting.weight_setter.set_weights") as mock_set_weights,
#     ):

#         mock_get_uids.return_value = [1, 2, 3, 4, 5]
#         mock_task_configs.return_value = [
#             TaskConfig(name="Task1", probability=0.5),
#             TaskConfig(name="Task2", probability=0.3),
#         ]
#         mock_reward_events.return_value = []

#         weight_setter.run_step()

#         mock_set_weights.assert_not_called()


# def test_set_weights():
#     with (
#         patch("prompting.weight_setter.settings.SUBTENSOR") as mock_subtensor,
#         patch("prompting.weight_setter.settings.WALLET") as mock_wallet,
#         patch("prompting.weight_setter.settings.NETUID") as mock_netuid,
#         patch("prompting.weight_setter.settings.METAGRAPH") as mock_metagraph,
#         patch("prompting.weight_setter.pd.DataFrame.to_csv") as mock_to_csv,
#     ):

#         weights = np.array([0.1, 0.2, 0.3, 0.4])
#         uids = np.array([1, 2, 3, 4])
#         mock_metagraph.uids = uids

#         set_weights(weights)

#         # Check if weights were processed and set
#         mock_subtensor.set_weights.assert_called_once()

#         # Check if weights were logged
#         if settings.LOG_WEIGHTS:
#             mock_to_csv.assert_called_once()
