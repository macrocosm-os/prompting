import asyncio
import copy
import sys
import threading
from traceback import print_exception

import numpy as np
import torch
from loguru import logger

from prompting.base.neuron import BaseNeuron
from prompting.rewards.reward import WeightedRewardEvent
from prompting.settings import settings
from prompting.utils.exceptions import MaxRetryError
from prompting.utils.logging import init_wandb


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        if settings.WANDB_ON:
            init_wandb(neuron="validator")
        self.latest_block = -1

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(settings.METAGRAPH.hotkeys)

        # Set up initial scoring weights for validation
        logger.info("Building validation weights.")
        self.scores = np.zeros(settings.METAGRAPH.n, dtype=np.float32)

        # Init sync with the network. Updates the metagraph.
        self.sync()

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None

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

        logger.info(f"Running validator with netuid: {settings.NETUID}")

        logger.info(f"Validator starting at block: {self.latest_block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                logger.info(f"step({self.step}) block({self.latest_block})")

                forward_timeout = settings.NEURON_FORWARD_MAX_TIME
                try:
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(self.forward())
                    logger.debug(f"Result of forward loop {result}")
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

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        logger.info("resync_metagraph()")
        # Sync the metagraph.
        settings.METAGRAPH.sync(subtensor=settings.SUBTENSOR)
