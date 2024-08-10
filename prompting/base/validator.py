import asyncio
import copy
import sys
import threading
from traceback import print_exception

import bittensor as bt
import numpy as np
import torch
from loguru import logger

from prompting import __spec_version__
from prompting.base.neuron import BaseNeuron
from prompting.settings import settings
from prompting.utils.exceptions import MaxRetryError
from prompting.utils.logging import init_wandb


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        init_wandb(neuron="validator")
        self.axon: bt.axon | None = None
        self.latest_block = -1

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(settings.METAGRAPH.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=settings.WALLET)
        logger.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        logger.info("Building validation weights.")
        self.scores = np.zeros(settings.METAGRAPH.n, dtype=np.float32)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Serve axon to enable external connections.
        self.axon = bt.axon(wallet=settings.WALLET, port=settings.AXON_PORT)
        if self.axon is not None:
            self._serve_axon()
        else:
            logger.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None

    def _serve_axon(self):
        """Serve axon to enable external connections"""
        validator_uid = settings.METAGRAPH.hotkeys.index(settings.WALLET.hotkey.ss58_address)
        logger.info(f"Serving validator IP of UID {validator_uid} to chain...")
        self.axon.serve(netuid=settings.NETUID, subtensor=settings.SUBTENSOR).start()

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """
        # Check that validator is registered on the network.
        self.sync()

        if not settings.NEURON_AXON_OFF:
            logger.info(f"Running validator {self.axon} with netuid: {settings.NETUID}")
        else:
            logger.info(f"Running validator with netuid: {settings.NETUID}")

        logger.info(f"Validator starting at block: {self.latest_block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                logger.info(f"step({self.step}) block({self.latest_block})")

                forward_timeout = settings.NEURON_FORWARD_MAX_TIME
                try:
                    task = self.loop.create_task(self.forward())
                    self.loop.run_until_complete(asyncio.wait_for(task, timeout=forward_timeout))
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"Out of memory error: {e}")
                    continue
                except MaxRetryError as e:
                    logger.error(f"MaxRetryError: {e}")
                    continue
                except asyncio.TimeoutError as e:
                    logger.error(
                        f"Forward timeout: Task execution exceeded {forward_timeout} seconds and was cancelled.: {e}"
                    )
                    continue
                except Exception as e:
                    logger.exception(e)

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            logger.success("Validator killed by keyboard interrupt.")
            sys.exit()

        # In case of unforeseen errors, the validator will log the error and quit
        except Exception as err:
            logger.error("Error during validation", str(err))
            logger.debug(print_exception(type(err), err, err.__traceback__))
            self.should_exit = True

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            logger.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            logger.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            logger.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            logger.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            logger.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            logger.debug("Stopped")

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if any(np.isnan(self.scores).flatten()):
            logger.warning(
                "Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = self.scores / np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        logger.debug("raw_weights", raw_weights)
        logger.debug("raw_weight_uids", settings.METAGRAPH.uids)
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=settings.METAGRAPH.uids,
            weights=raw_weights,
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
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        logger.debug("uint_weights", uint_weights)
        logger.debug("uint_uids", uint_uids)

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

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        logger.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(settings.METAGRAPH)

        # Sync the metagraph.
        settings.METAGRAPH.sync(subtensor=settings.SUBTENSOR)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == settings.METAGRAPH.axons:
            return

        logger.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
        logger.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != settings.METAGRAPH.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(settings.METAGRAPH.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = np.zeros((settings.METAGRAPH.n))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(settings.METAGRAPH.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: list[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if any(np.isnan(rewards).flatten()):
            # if
            logger.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        step_rewards = np.copy(self.scores)
        step_rewards[uids] = rewards
        logger.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha = settings.NEURON_MOVING_AVERAGE_ALPHA
        self.scores = alpha * step_rewards + (1 - alpha) * self.scores
        self.scores = np.clip(self.scores - settings.NEURON_DECAY_ALPHA, 0, 1)
        logger.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        logger.info("Saving validator state.")

        # Save the state of the validator to file.
        np.savez(settings.SAVE_PATH + "/state.npz", step=self.step, scores=self.scores, hotkeys=self.hotkeys)

    def load_state(self):
        """Loads the state of the validator from a file."""
        logger.info("Loading validator state.")

        # Load the state of the validator from file.
        state = np.load(settings.SAVE_PATH + "/state.npz")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
