import os
import torch
import dotenv
from loguru import logger

# from pydantic import BaseModel, Field
# from typing import ClassVar

# class Settings(BaseModel):
#     TASK_P: ClassVar[list[float]] = Field([0.5, 0.5], description="TODO: Dynamically load based on number of tasks")
if not dotenv.load_dotenv():
    logger.warning("No .env file found")

test = os.environ.get("TEST", False)
WALLET_NAME = os.environ.get("WALLET_NAME")
HOTKEY = os.environ.get("HOTKEY")

assert WALLET_NAME and HOTKEY, "You must provide you wallet and hotkey name in the .env file!"

# Set up path for storage and create it if it doesn't exist
SAVE_PATH = "./storage"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

AXON_PORT = os.environ.get("AXON_PORT")
# Constants
TASK_P = [0.5, 0.5]  # TODO: Dynamically load based on number of tasks
SUBTENSOR_NETWORK = "test" if test else None
NETUID = 61 if test else 1
NEURON_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NEURON_GPUS = 1
NEURON_LLM_MAX_ALLOWED_MEMORY_IN_GB = 24 if test else 70
NEURON_EPOCH_LENGTH = 100
MOCK = False
NEURON_EVENTS_RETENTION_SIZE = "2 GB"
NEURON_DONT_SAVE_EVENTS = False
NEURON_LOG_FULL = False
NO_BACKGROUND_THREAD = True
WANDB_OFFLINE = False
WANDB_NOTES = ""

NEURON_MODEL_ID_MINER = "gpt-3.5-turbo-1106"
NEURON_MODEL_ID_VALIDATOR = "casperhansen/llama-3-8b-instruct-awq" if test else "casperhansen/llama-3-70b-instruct-awq"
BLACKLIST_FORCE_VALIDATOR_PERMIT = False
BLACKLIST_ALLOW_NON_REGISTERED = False
NEURON_SYSTEM_PROMPT = "You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know."
NEURON_MAX_TOKENS = 256
NEURON_TEMPERATURE = 0.7
NEURON_TOP_K = 50
NEURON_TOP_P = 0.95
NEURON_STOP_ON_FORWARD_EXCEPTION = False
NEURON_SHOULD_FORCE_MODEL_LOADING = False
WANDB_ON = False
WANDB_ENTITY = "opentensor-dev"
WANDB_PROJECT_NAME_MINER = "alpha-miners"
WANDB_PROJECT_NAME_VALIDATOR = "alpha-validators"
NEURON_STREAMING_BATCH_SIZE = 12
NEURON_TIMEOUT = 15
NEURON_NUM_CONCURRENT_FORWARDS = 1
NEURON_SAMPLE_SIZE = 100
NEURON_DISABLE_SET_WEIGHTS = False
NEURON_MOVING_AVERAGE_ALPHA = 0.1
NEURON_DECAY_ALPHA = 0.001
NEURON_AXON_OFF = False
NEURON_VPERMIT_TAO_LIMIT = 4096
NEURON_QUERY_UNIQUE_COLDKEYS = False
NEURON_QUERY_UNIQUE_IPS = False
NEURON_FORWARD_MAX_TIME = 120
