import os
from functools import cached_property
from typing import Any, Literal, Optional

import bittensor as bt
import dotenv
from loguru import logger
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mode: Literal["miner", "validator", "mock"]
    MOCK: bool = False
    NO_BACKGROUND_THREAD: bool = True
    SAVE_PATH: Optional[str] = Field("./storage", env="SAVE_PATH")
    model_config = {"frozen": True, "arbitrary_types_allowed": False}

    # Class variables for singleton.
    _instance: Optional["Settings"] = None
    _instance_mode: Optional[str] = None

    @classmethod
    def load_env_file(cls, mode: Literal["miner", "validator", "mock"]):
        """Load the appropriate .env file based on the mode."""
        if mode == "miner":
            dotenv_file = ".env.miner"
        elif mode == "validator":
            dotenv_file = ".env.validator"
        # For mock testing, still make validator env vars available where possible.
        elif mode == "mock":
            dotenv_file = ".env.validator"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if dotenv_file:
            if not dotenv.load_dotenv(dotenv.find_dotenv(filename=dotenv_file)):
                logger.warning(
                    f"No {dotenv_file} file found. The use of args when running a {mode} will be deprecated "
                    "in the near future."
                )

    @classmethod
    def load(cls, mode: Literal["miner", "validator", "mock"]) -> "Settings":
        """Load or retrieve the Settings instance based on the mode."""
        if cls._instance is not None and cls._instance_mode == mode:
            return cls._instance
        else:
            cls.load_env_file(mode)
            cls._instance = cls(mode=mode)
            cls._instance_mode = mode
            return cls._instance

    @model_validator(mode="before")
    def complete_settings(cls, values: dict[str, Any]) -> dict[str, Any]:
        mode = values["mode"]
        netuid = values.get("NETUID", 61)

        if netuid is None:
            raise ValueError("NETUID must be specified")
        values["TEST"] = netuid != 1
        if values.get("TEST_MINER_IDS"):
            values["TEST_MINER_IDS"] = str(values["TEST_MINER_IDS"]).split(",")
        if mode == "mock":
            values["MOCK"] = True
            values["NEURON_DEVICE"] = "cpu"
            logger.info("Running in mock mode. Bittensor objects will not be initialized.")
            return values

        # load slow packages only if not in mock mode
        import torch

        if not values.get("NEURON_DEVICE"):
            values["NEURON_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure SAVE_PATH exists.
        save_path = values.get("SAVE_PATH", "./storage")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if values.get("SN19_API_KEY") is None or values.get("SN19_API_URL") is None:
            logger.warning(
                "It is strongly recommended to provide an SN19 API KEY + URL to avoid incurring OpenAI API costs."
            )
        if mode == "validator":
            if values.get("OPENAI_API_KEY") is None:
                raise Exception(
                    "You must provide an OpenAI API key as a backup. It is recommended to also provide an SN19 API key + url to avoid incurring API costs."
                )
            if values.get("SCORING_ADMIN_KEY") is None:
                raise Exception("You must provide an admin key to access the API.")
            if values.get("PROXY_URL") is None:
                logger.warning(
                    "You must provide a proxy URL to use the DuckDuckGo API - your vtrust might decrease if no DDG URL is provided."
                )
        return values

    @cached_property
    def WALLET(self):
        wallet_name = self.WALLET_NAME  # or config().wallet.name
        hotkey = self.HOTKEY  # or config().wallet.hotkey
        logger.info(f"Instantiating wallet with name: {wallet_name}, hotkey: {hotkey}")
        return bt.wallet(name=wallet_name, hotkey=hotkey)

    @cached_property
    def SUBTENSOR(self) -> bt.subtensor:
        subtensor_network = self.SUBTENSOR_NETWORK or os.environ.get("SUBTENSOR_NETWORK", "local")
        # bt_config = config()
        if subtensor_network.lower() == "local":
            subtensor_network = os.environ.get("SUBTENSOR_CHAIN_ENDPOINT")  # bt_config.subtensor.chain_endpoint or
        else:
            subtensor_network = subtensor_network.lower()  # bt_config.subtensor.network or
        logger.info(f"Instantiating subtensor with network: {subtensor_network}")
        return bt.subtensor(network=subtensor_network)

    @cached_property
    def METAGRAPH(self) -> bt.metagraph:
        logger.info(f"Instantiating metagraph with NETUID: {self.NETUID}")
        return self.SUBTENSOR.metagraph(netuid=self.NETUID)

    @cached_property
    def DENDRITE(self) -> bt.dendrite:
        logger.info(f"Instantiating dendrite with wallet: {self.WALLET}")
        return bt.dendrite(wallet=self.WALLET)


logger.info("Settings class instantiated.")
settings: Optional[Settings] = None
try:
    settings: Optional[Settings] = Settings.load(mode="mock")
    pass
except Exception as e:
    logger.exception(f"Error loading settings: {e}")
    settings = None
logger.info("Settings loaded.")
