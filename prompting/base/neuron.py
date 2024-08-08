import sys
import bittensor as bt
from loguru import logger
from abc import ABC, abstractmethod

# Sync calls set weights and also resyncs the metagraph.
from prompting.utils.config import config
from prompting.utils.misc import ttl_get_block
from prompting import __spec_version__ as spec_version

from prompting import settings


class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    @classmethod
    def _config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = spec_version

    @property
    def block(self):
        self._block = ttl_get_block(self)
        self.latest_block = self._block or -1
        return self._block

    def __init__(self, config=None):
        self.config = self._config()

        # If a gpu is required, set the device to cuda:N (e.g. cuda:0)
        self.device = settings.NEURON_DEVICE

        # Log the configuration for reference.
        logger.info(self.config)

        # Build Bittensor objects
        # These are core Bittensor classes to interact with the network.
        logger.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(settings.NETUID)

        logger.info(f"Wallet: {self.wallet}")
        logger.info(f"Subtensor: {self.subtensor}")
        logger.info(f"Metagraph: {self.metagraph}")

        # Check if the miner is registered on the Bittensor network before proceeding further.
        self.check_registered()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info(f"Running neuron on subnet: {settings.NETUID} with uid {self.uid}")
        self.step = 0

    @abstractmethod
    def forward(self, synapse: bt.Synapse) -> bt.Synapse: ...

    @abstractmethod
    def run(self): ...

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        # Ensure miner or validator hotkey is still registered on the network.
        logger.info("Syncing neuron...")
        self.check_registered()

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            self.set_weights()

        # Always save state.
        self.save_state()

    def check_registered(self):
        # --- Check for registration.
        if not self.subtensor.is_hotkey_registered(
            netuid=settings.NETUID,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            logger.error(
                f"Wallet: {self.wallet} is not registered on netuid {settings.NETUID}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit()

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return (
            self.subtensor.get_current_block() - self.metagraph.last_update[self.uid]
        ) > settings.NEURON_EPOCH_LENGTH

    def should_set_weights(self) -> bool:
        # Don't set weights on initialization.
        if self.step == 0:
            return False

        # Check if enough epoch blocks have elapsed since the last epoch.
        if settings.NEURON_DISABLE_SET_WEIGHTS:
            return False

        # If neuron has validator permit we assume its running the validator code. If it is a dual permit neuron then we check that it also has a set_weights method (only true if it is running validator neuron)
        if not self.metagraph.validator_permit[self.uid] or not hasattr(self, "set_weights"):
            return False

        # Define appropriate logic for when set weights.
        return (self.block - self.metagraph.last_update[self.uid]) > settings.NEURON_EPOCH_LENGTH

    def save_state(self):
        pass

    def load_state(self):
        logger.debug(
            "load_state() not implemented for this neuron. You can implement this function to load model checkpoints or other useful data."
        )
