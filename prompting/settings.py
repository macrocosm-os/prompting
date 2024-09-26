import os
import torch
import dotenv
from loguru import logger
import bittensor as bt
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from typing import Any, Literal, Optional, List
from prompting.utils.config import config


def load_env_file(mode: Literal["miner", "validator", "mock"]):
    """Load the appropriate .env file based on the mode."""
    if mode == "miner":
        dotenv_file = ".env.miner"
    elif mode == "validator":
        dotenv_file = ".env.validator"
    elif mode == "mock":
        dotenv_file = None
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if dotenv_file:
        if not dotenv.load_dotenv(dotenv.find_dotenv(filename=dotenv_file)):
            logger.warning(
                f"No {dotenv_file} file found. The use of args when running a {mode} will be deprecated "
                "in the near future."
            )


class Settings(BaseSettings):
    mode: Literal["miner", "validator", "mock"]
    MOCK: bool = False
    NO_BACKGROUND_THREAD: bool = True
    SAVE_PATH: Optional[str] = Field("./storage", env="SAVE_PATH")

    # W&B.
    WANDB_ON: bool = Field(True, env="WANDB_ON")
    WANDB_ENTITY: Optional[str] = Field("macrocosmos", env="WANDB_ENTITY")
    WANDB_PROJECT_NAME: Optional[str] = Field("prompting-validators", env="WANDB_PROJECT_NAME")
    WANDB_RUN_STEP_LENGTH: int = Field(100, env="WANDB_RUN_STEP_LENGTH")
    WANDB_API_KEY: Optional[str] = Field(None, env="WANDB_API_KEY")
    WANDB_OFFLINE: bool = Field(False, env="WANDB_OFFLINE")
    WANDB_NOTES: str = Field("", env="WANDB_NOTES")

    # Neuron.
    NEURON_EPOCH_LENGTH: int = Field(1, env="NEURON_EPOCH_LENGTH")
    NEURON_DEVICE: str = Field("cuda" if torch.cuda.is_available() else "cpu", env="NEURON_DEVICE")
    NEURON_GPUS: int = Field(1, env="NEURON_GPUS")

    # Logging.
    LOGGING_DONT_SAVE_EVENTS: bool = Field(False, env="LOGGING_DONT_SAVE_EVENTS")
    LOG_WEIGHTS: bool = Field(False, env="LOG_WEIGHTS")

    # Neuron parameters.
    NEURON_TIMEOUT: int = Field(15, env="NEURON_TIMEOUT")
    NEURON_DISABLE_SET_WEIGHTS: bool = Field(False, env="NEURON_DISABLE_SET_WEIGHTS")
    NEURON_MOVING_AVERAGE_ALPHA: float = Field(0.1, env="NEURON_MOVING_AVERAGE_ALPHA")
    NEURON_DECAY_ALPHA: float = Field(0.001, env="NEURON_DECAY_ALPHA")
    NEURON_AXON_OFF: bool = Field(False, env="NEURON_AXON_OFF")
    NEURON_VPERMIT_TAO_LIMIT: int = Field(4096, env="NEURON_VPERMIT_TAO_LIMIT")
    NEURON_QUERY_UNIQUE_COLDKEYS: bool = Field(False, env="NEURON_QUERY_UNIQUE_COLDKEYS")
    NEURON_QUERY_UNIQUE_IPS: bool = Field(False, env="NEURON_QUERY_UNIQUE_IPS")
    NEURON_FORWARD_MAX_TIME: int = Field(240, env="NEURON_FORWARD_MAX_TIME")
    NEURON_MAX_TOKENS: int = Field(512, env="NEURON_MAX_TOKENS")

    # Organic.
    ORGANIC_TIMEOUT: int = Field(30, env="ORGANIC_TIMEOUT")
    ORGANIC_SAMPLE_SIZE: int = Field(5, env="ORGANIC_SAMPLE_SIZE")
    ORGANIC_REUSE_RESPONSE_DISABLED: bool = Field(False, env="ORGANIC_REUSE_RESPONSE_DISABLED")
    ORGANIC_REFERENCE_MAX_TOKENS: int = Field(1024, env="ORGANIC_REFERENCE_MAX_TOKENS")
    ORGANIC_SYNTH_REWARD_SCALE: float = Field(1.0, env="ORGANIC_SYNTH_REWARD_SCALE")
    ORGANIC_SET_WEIGHTS_ENABLED: bool = Field(True, env="ORGANIC_SET_WEIGHTS_ENABLED")
    ORGANIC_DISABLED: bool = Field(False, env="ORGANIC_DISABLED")
    ORGANIC_TRIGGER_FREQUENCY: int = Field(120, env="ORGANIC_TRIGGER_FREQUENCY")
    ORGANIC_TRIGGER_FREQUENCY_MIN: int = Field(5, env="ORGANIC_TRIGGER_FREQUENCY_MIN")
    ORGANIC_TRIGGER: str = Field("seconds", env="ORGANIC_TRIGGER")
    ORGANIC_SCALING_FACTOR: int = Field(1, env="ORGANIC_SCALING_FACTOR")
    HF_TOKEN: Optional[str] = Field(None, env="HF_TOKEN")

    # Additional Fields.
    NETUID: Optional[int] = Field(61, env="NETUID")
    TEST: bool = False
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    WALLET_NAME: Optional[str] = Field(None, env="WALLET_NAME")
    HOTKEY: Optional[str] = Field(None, env="HOTKEY")
    AXON_PORT: Optional[int] = Field(None, env="AXON_PORT")
    ORGANIC_WHITELIST_HOTKEY: Optional[str] = Field(
        "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3", env="ORGANIC_WHITELIST_HOTKEY"
    )
    TEST_MINER_IDS: Optional[List[int]] = Field(None, env="TEST_MINER_IDS")
    SUBTENSOR_NETWORK: Optional[str] = Field(None, env="SUBTENSOR_NETWORK")
    NEURON_LLM_MAX_ALLOWED_MEMORY_IN_GB: int = Field(62, env="MAX_ALLOWED_VRAM_GB")
    LLM_MAX_MODEL_LEN: int = Field(4096, env="LLM_MAX_MODEL_LEN")
    NEURON_MODEL_ID_VALIDATOR: str = Field(
        "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", env="LLM_MODEL"
    )
    MINER_LLM_MODEL: Optional[str] = Field(None, env="MINER_LLM_MODEL")
    LLM_MODEL_RAM: float = Field(70, env="LLM_MODEL_RAM")

    # Bittensor Objects.
    WALLET: Optional[bt.wallet] = None
    SUBTENSOR: Optional[bt.subtensor] = None
    METAGRAPH: Optional[bt.metagraph] = None
    DENDRITE: Optional[bt.dendrite] = None

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def complete_settings(cls, settings: "Settings") -> "Settings":
        mode = settings.mode
        netuid = settings.NETUID

        if netuid is None:
            raise ValueError("NETUID must be specified")

        # Collect updates to avoid mutating the frozen instance.
        updates: dict[str, Any] = {}

        updates["TEST"] = netuid != 1

        if mode == "mock":
            updates["MOCK"] = True
            logger.info("Running in mock mode. Bittensor objects will not be initialized.")
            return settings.model_copy(update=updates)

        bt_config = config()
        wallet_name = settings.WALLET_NAME or bt_config.wallet.name
        hotkey = settings.HOTKEY or bt_config.wallet.hotkey
        updates["WALLET_NAME"] = wallet_name
        updates["HOTKEY"] = hotkey

        logger.info(
            f"Instantiating bittensor objects with NETUID: {netuid}, WALLET_NAME: {wallet_name}, HOTKEY: {hotkey}"
        )

        subtensor_network = settings.SUBTENSOR_NETWORK or os.environ.get("SUBTENSOR_NETWORK", "local")
        if subtensor_network.lower() == "local":
            subtensor_network = bt_config.subtensor.chain_endpoint or os.environ.get("SUBTENSOR_CHAIN_ENDPOINT")
        else:
            subtensor_network = bt_config.subtensor.network or subtensor_network.lower()
        updates["SUBTENSOR_NETWORK"] = subtensor_network

        # Initialize Bittensor Objects.
        wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
        updates["WALLET"] = wallet
        subtensor = bt.subtensor(network=subtensor_network)
        updates["SUBTENSOR"] = subtensor
        metagraph = bt.metagraph(netuid=netuid, network=subtensor_network, sync=True, lite=True)
        updates["METAGRAPH"] = metagraph
        updates["DENDRITE"] = bt.dendrite(wallet=wallet)

        logger.info(
            f"Bittensor objects instantiated... WALLET: {wallet}, SUBTENSOR: {subtensor}, METAGRAPH: {metagraph}"
        )

        # Parse TEST_MINER_IDS if provided.
        test_miner_ids = settings.TEST_MINER_IDS
        if test_miner_ids and isinstance(test_miner_ids, str):
            updates["TEST_MINER_IDS"] = [int(miner_id) for miner_id in test_miner_ids.split(",")]

        save_path = settings.SAVE_PATH
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        return settings.model_copy(update=updates)


def load_settings(mode: str) -> Settings:
    load_env_file(mode)
    settings = Settings(mode=mode)
    return settings


settings: Settings | None = None
