from loguru import logger
import bittensor as bt
import numpy as np
import os
import pandas as pd

from prompting import __spec_version__
from prompting.settings import settings
from prompting.utils.uids import get_uids
from prompting.utils.misc import ttl_get_block
from prompting.base.loop_runner import AsyncLoopRunner
from prompting import mutable_globals
from prompting.rewards.reward import WeightedRewardEvent
from prompting.tasks.task_registry import TaskRegistry, TaskConfig
from prompting.tasks.inference import InferenceTask
from prompting.llms.model_zoo import ModelZoo

PAST_WEIGHTS: list[np.ndarray] = []
WEIGHTS_HISTORY_LENGTH = 24


def apply_reward_func(raw_rewards, p=0.5):
    """Apply the reward function to the raw rewards. P adjusts the steepness of the function - p = 0.5 leaves
    the rewards unchanged, p < 0.5 makes the function more linear (at p=0 all miners with positives reward values get the same reward),
    p > 0.5 makes the function more exponential (winner takes all).
    """
    exponent = (p**6.64385619) * 100  # 6.64385619 = ln(100)/ln(2) -> this way if p=0.5, the exponent is exatly 1
    positive_rewards = np.clip(raw_rewards, 1e-10, np.inf)
    normalised_rewards = positive_rewards / np.max(positive_rewards)
    post_func_rewards = normalised_rewards**exponent
    all_rewards = post_func_rewards
    all_rewards[raw_rewards <= 0] = 0
    return all_rewards / (np.sum(all_rewards) + 1e-10)


def set_weights(weights: np.ndarray, step: int = 0):
    """
    Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
    """

    # Check if self.scores contains any NaN values and log a warning if it does.
    if any(np.isnan(weights).flatten()):
        logger.warning(
            f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions. Scores: {weights}"
        )

    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = weights / np.linalg.norm(weights, ord=1, axis=0, keepdims=True)
    PAST_WEIGHTS.append(raw_weights)
    if len(PAST_WEIGHTS) > WEIGHTS_HISTORY_LENGTH:
        PAST_WEIGHTS.pop(0)
    averaged_weights = np.average(np.array(PAST_WEIGHTS), axis=0)

    logger.debug("raw_weights", averaged_weights)
    logger.debug("raw_weight_uids", settings.METAGRAPH.uids)
    # Process the raw weights to final_weights via subtensor limitations.
    (
        processed_weight_uids,
        processed_weights,
    ) = bt.utils.weight_utils.process_weights_for_netuid(
        uids=settings.METAGRAPH.uids,
        weights=averaged_weights,
        netuid=settings.NETUID,
        subtensor=settings.SUBTENSOR,
        metagraph=settings.METAGRAPH,
    )
    logger.debug("processed_weights", processed_weights)
    logger.debug("processed_weight_uids", processed_weight_uids)

    # Convert to uint16 weights and uids.
    (
        uint_uids,
        uint_weights,
    ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(uids=processed_weight_uids, weights=processed_weights)
    logger.debug("uint_weights", uint_weights)
    logger.debug("uint_uids", uint_uids)

    # Create a dataframe from weights and uids and save it as a csv file, with the current step as the filename.
    if settings.LOG_WEIGHTS:
        weights_df = pd.DataFrame(
            {
                "step": step,
                "uids": uint_uids,
                "weights": uint_weights,
                "raw_weights": raw_weights.flatten(),
                "block": ttl_get_block(),
            }
        )
        step_filename = "weights.csv"
        file_exists = os.path.isfile(step_filename)
        # Append to the file if it exists, otherwise write a new file.
        weights_df.to_csv(step_filename, mode="a", index=False, header=not file_exists)

    if settings.NEURON_DISABLE_SET_WEIGHTS:
        logger.debug(f"Set weights disabled: {settings.NEURON_DISABLE_SET_WEIGHTS}")
        return

    # Set the weights on chain via our subtensor connection.
    result = settings.SUBTENSOR.set_weights(
        wallet=settings.WALLET,
        netuid=settings.NETUID,
        uids=uint_uids,
        weights=uint_weights,
        wait_for_finalization=False,
        wait_for_inclusion=False,
        version_key=__spec_version__,
    )

    if result is True:
        logger.info("set_weights on chain successfully!")
    else:
        logger.error("set_weights failed")


class WeightSetter(AsyncLoopRunner):
    """The weight setter looks at RewardEvents in the reward_events queue and sets the weights of the miners accordingly."""

    sync: bool = True
    interval: int = 10

    async def run_step(self):
        if len(mutable_globals.reward_events) == 0:
            logger.warning("No reward events in queue, skipping weight setting...")
            return
        logger.debug(f"Found {len(mutable_globals.reward_events)} reward events in queue")

        # reward_events is a list of lists of WeightedRewardEvents - the 'sublists' each contain the multiple reward events for a single task
        mutable_globals.reward_events: list[list[WeightedRewardEvent]] = (
            mutable_globals.reward_events
        )  # to get correct typehinting

        reward_dict = {uid: 0 for uid in get_uids(sampling_mode="all")}
        # miner_rewards is a dictionary that separates each task config into a dictionary of uids with their rewards
        miner_rewards: dict[TaskConfig, dict[int, float]] = {
            config: {uid: 0 for uid in get_uids(sampling_mode="all")} for config in TaskRegistry.task_configs
        }
        logger.debug(f"Miner rewards before processing: {miner_rewards}")

        inference_events: list[WeightedRewardEvent] = []
        for reward_events in mutable_globals.reward_events:
            for reward_event in reward_events:
                if np.sum(reward_event.rewards) > 0:
                    logger.debug("Identified positive reward event")
                task_config = TaskRegistry.get_task_config(reward_event.task)

                # inference task uses a different reward model
                if task_config.task == InferenceTask:
                    inference_events.append(reward_event)
                    continue

                # give each uid the reward they received
                for uid, reward in zip(reward_event.uids, reward_event.rewards):
                    if reward > 0:
                        logger.debug(f"Adding reward {reward} to uid {uid} for task {task_config.task.__name__}")
                    miner_rewards[task_config][uid] += reward * reward_event.weight
        logger.debug(f"Miner rewards after processing: {miner_rewards}")

        for inference_event in inference_events:
            for uid, reward in zip(inference_event.uids, inference_event.rewards):
                llm_model = inference_event.task.llm_model_id

                model_specific_reward = ModelZoo.get_model_by_id(llm_model).reward if llm_model else 1
                miner_rewards[TaskRegistry.get_task_config(InferenceTask)][uid] += reward * model_specific_reward

        for task_config, rewards in miner_rewards.items():
            r = np.array(list(rewards.values()))
            logger.debug(f"Rewards for task {task_config.task.__name__}: {r}")
            u = np.array(list(rewards.keys()))
            if task_config.task == InferenceTask:
                processed_rewards = r / (np.sum(r) + 1e-10)
            else:
                processed_rewards = apply_reward_func(raw_rewards=r, p=settings.REWARD_STEEPNESS)
            processed_rewards *= task_config.probability
            # update reward dict
            for uid, reward in zip(u, processed_rewards):
                reward_dict[uid] += reward
        logger.debug(f"Final reward dict: {reward_dict}")

        # set weights on chain
        set_weights(np.array(list(reward_dict.values())), step=self.step)
        return reward_dict


weight_setter = WeightSetter()
