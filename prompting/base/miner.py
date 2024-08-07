import time
import wandb
import prompting
import threading
import bittensor as bt
from datetime import datetime
from prompting.base.protocol import StreamPromptingSynapse
from prompting.base.neuron import BaseNeuron
from traceback import print_exception
from prompting import settings
from loguru import logger
from pydantic import BaseModel, model_validator, ConfigDict
from typing import Tuple


class BaseStreamMinerNeuron(BaseModel, BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    step: int = 0
    wallet: bt.wallet | None = None
    axon: bt.axon | None = None
    subtensor: bt.subtensor | None = None
    metagraph: bt.metagraph | None = None
    should_exit: bool = False
    is_running: bool = False
    thread: threading.Thread = None
    uid: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def attach_axon(self) -> "BaseStreamMinerNeuron":
        # note that this initialization has to happen in the validator because the objects
        # are not picklable and because pydantic deepcopies things it breaks
        self.wallet = bt.wallet(name=settings.WALLET_NAME, hotkey=settings.HOTKEY)
        self.axon = bt.axon(wallet=self.wallet, port=settings.AXON_PORT)
        logger.info("Attaching axon")
        self.axon.attach(
            forward_fn=self._forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        self.subtensor = bt.subtensor(network=settings.SUBTENSOR_NETWORK)
        self.metagraph = bt.metagraph(netuid=settings.NETUID, network=settings.SUBTENSOR_NETWORK, sync=True, lite=False)
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info(f"Axon created: {self.axon}; running on uid: {self.uid}")
        self.axon.serve(netuid=settings.NETUID, subtensor=self.subtensor)
        return self

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        logger.info(f"Serving miner axon {self.axon} with netuid: {settings.NETUID}")

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()

        logger.info(f"Miner starting at block: {self.subtensor.get_current_block()}")
        last_update_block = 0
        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while self.subtensor.get_current_block() - last_update_block < settings.NEURON_EPOCH_LENGTH:
                    # Wait before checking again.
                    time.sleep(1)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph and potentially set weights.
                self.sync()
                last_update_block = self.subtensor.get_current_block()
                self.step += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            logger.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as err:
            logger.error("Error during mining", str(err))
            logger.debug(print_exception(type(err), err, err.__traceback__))
            self.should_exit = True

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            logger.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            logger.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            logger.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            logger.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        logger.info("resync_metagraph()")

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

    def _forward(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        """
        A wrapper method around the `forward` method that will be defined by the subclass.

        This method acts as an intermediary layer to perform pre-processing before calling the
        actual `forward` method implemented in the subclass. Specifically, it checks whether a
        prompt is in cache to avoid reprocessing recent requests. If the prompt is not in the
        cache, the subclass `forward` method is called.

        Args:
            synapse (StreamPromptingSynapse): The incoming request object encapsulating the details of the request.

        Returns:
            StreamPromptingSynapse: The response object to be sent back in reply to the incoming request, essentially
            the filled synapse request object.

        Raises:
            ValueError: If the prompt is found in the cache indicating it was sent recently.

        Example:
            This method is not meant to be called directly but is invoked internally when a request
            is received, and it subsequently calls the `forward` method of the subclass.
        """
        self.step += 1
        logger.info("Calling self._forward in BaseStreamMinerNeuron")
        return self.forward(synapse=synapse)

    async def blacklist(self, synapse: StreamPromptingSynapse) -> Tuple[bool, str]:
        # WARNING: The typehint must remain Tuple[bool, str] to avoid runtime errors. YOU
        # CANNOT change to tuple[bool, str]!!!
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (StreamPromptingSynapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            logger.trace(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        logger.trace(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: StreamPromptingSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (StreamPromptingSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)  # Get the caller index.
        priority = float(self.metagraph.S[caller_uid])  # Return the stake as the priority.
        logger.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority)
        # priority = 1.0
        return priority

    def init_wandb(self):
        logger.info("Initializing wandb...")

        uid = f"uid_{self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)}"
        net_uid = f"netuid_{settings.NETUID}"
        tags = [
            self.wallet.hotkey.ss58_address,
            net_uid,
            f"uid_{uid}",
            prompting.__version__,
            str(prompting.__spec_version__),
        ]

        # Add uid, netuid and timestamp to run name
        run_name_tags = [
            uid,
            net_uid,
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        ]

        # Compose run name
        run_name = "_".join(run_name_tags)

        # inits wandb in case it hasn't been inited yet
        self.wandb_run = wandb.init(
            name=run_name,
            project=settings.WANDB_PROJECT_NAME_MINER,
            entity=settings.WANDB_ENTITY,
            config=self.config,  # TODO: Check whether we really want this
            mode="online" if settings.WANDB_ON else "offline",
            tags=tags,
        )

    def log_event(
        self,
        synapse: StreamPromptingSynapse,
        timing: float,
        messages,
        accumulated_chunks: list[str] = [],
        accumulated_chunks_timings: list[float] = [],
        extra_info: dict = {},
    ):
        if not getattr(self, "wandb_run", None):
            self.init_wandb()

        dendrite_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        step_log = {
            "epoch_time": timing,
            # TODO: add block to logs in the future in a way that doesn't impact performance
            # "block": self.block,
            "messages": messages,
            "accumulated_chunks": accumulated_chunks,
            "accumulated_chunks_timings": accumulated_chunks_timings,
            "validator_uid": dendrite_uid,
            "validator_ip": synapse.dendrite.ip,
            "validator_coldkey": self.metagraph.coldkeys[dendrite_uid],
            "validator_hotkey": self.metagraph.hotkeys[dendrite_uid],
            "validator_stake": self.metagraph.S[dendrite_uid].item(),
            "validator_trust": self.metagraph.T[dendrite_uid].item(),
            "validator_incentive": self.metagraph.I[dendrite_uid].item(),
            "validator_consensus": self.metagraph.C[dendrite_uid].item(),
            "validator_dividends": self.metagraph.D[dendrite_uid].item(),
            "miner_stake": self.metagraph.S[self.uid].item(),
            "miner_trust": self.metagraph.T[self.uid].item(),
            "miner_incentive": self.metagraph.I[self.uid].item(),
            "miner_consensus": self.metagraph.C[self.uid].item(),
            "miner_dividends": self.metagraph.D[self.uid].item(),
            **extra_info,
        }

        logger.info("Logging event to wandb...", step_log)
        wandb.log(step_log)

    def log_status(self):
        m = self.metagraph
        logger.info(
            f"Miner running:: network: {self.subtensor.network} | step: {self.step} | uid: {self.uid} | trust: {m.trust[self.uid]:.3f} | emission {m.emission[self.uid]:.3f}"
        )
