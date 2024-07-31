# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import torch
import argparse
import bittensor as bt
import logging
from prompting.tasks import TASKS

from bittensor.btlogging.defines import BITTENSOR_LOGGER_NAME

logger = logging.getLogger(BITTENSOR_LOGGER_NAME)


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    bt.logging.info(f"Logging path: {full_path}")
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        event_handler = logging.FileHandler(
            os.path.join(config.neuron.full_path, "events.log")
        )
        event_handler.setLevel(38)  # Custom level
        formatter = logging.Formatter("{asctime} | {levelname} | {message}", style="{")
        event_handler.setFormatter(formatter)
        logger.addHandler(event_handler)
        logging.addLevelName(38, "EVENTS")


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--neuron.gpus",
        type=int,
        help="The number of visible GPUs to be considered in the llm initialization. This parameter currently reflects on the property `tensor_parallel_size` of vllm",
        default=1,
    )

    parser.add_argument(
        "--neuron.llm_max_allowed_memory_in_gb",
        type=int,
        help="The max gpu memory utilization set for initializing the model. This parameter currently reflects on the property `gpu_memory_utilization` of vllm",
        default=62,
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock neuron and all network components.",
        default=False,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default="2 GB",
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--neuron.log_full",
        action="store_true",
        help="If set, logs more information.",
        default=False,
    )

    parser.add_argument(
        "--no_background_thread",
        action="store_true",
        help="If set, we dont run the neuron in a background thread.",
        default=True,
    )

    parser.add_argument(
        "--wandb.off", action="store_true", help="Turn off wandb.", default=False
    )

    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        help="Runs wandb in offline mode.",
        default=False,
    )

    parser.add_argument(
        "--wandb.notes",
        type=str,
        help="Notes to add to the wandb run.",
        default="",
    )


def add_miner_args(cls, parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="miner",
    )

    parser.add_argument(
        "--neuron.model_id",
        type=str,
        help="The model to use for the validator.",
        default="gpt-3.5-turbo-1106",
    )

    parser.add_argument(
        "--neuron.load_in_8bit",
        type=str,
        default=False,
        help="Load quantized model in 8 bits. Note that this parameter only applies to hugging face miners.",
    )

    parser.add_argument(
        "--neuron.load_in_4bit",
        type=str,
        default=False,
        help="Load quantized model in 4 bits. Note that this parameter only applies to hugging face miners.",
    )

    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        help="If set, we will force incoming requests to have a permit.",
        default=False,
    )

    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        help="If set, miners will accept queries from non registered entities. (Dangerous!)",
        default=False,
    )

    parser.add_argument(
        "--neuron.system_prompt",
        type=str,
        help="The system prompt to use for the miner.",
        default="You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know.",
    )

    parser.add_argument(
        "--neuron.max_tokens",
        type=int,
        default=256,
        help="The maximum number of tokens to generate in the completion.",
    )

    parser.add_argument(
        "--neuron.temperature",
        type=float,
        default=0.7,
        help="Sampling temperature to use, between 0 and 2.",
    )

    parser.add_argument(
        "--neuron.top_k",
        type=float,
        default=50,
        help="Nucleus sampling parameter, top_p probability mass.",
    )

    parser.add_argument(
        "--neuron.top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter, top_p probability mass.",
    )

    parser.add_argument(
        "--neuron.stop_on_forward_exception",
        type=bool,
        default=False,
        help="Set miner to stop on forward exception.",
    )

    parser.add_argument(
        "--neuron.should_force_model_loading",
        type=bool,
        default=False,
        help="Force model loading independent of mock flag.",
    )

    parser.add_argument(
        "--wandb.on",
        type=bool,
        default=False,
        help="Enable wandb logging.",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="opentensor-dev",
        help="Wandb entity to log to.",
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="alpha-miners",
        help="Wandb project to log to.",
    )

    parser.add_argument(
        "--neuron.streaming_batch_size",
        type=int,
        default=12,
        help="Batch size in tokens for streaming forward calls.",
    )


def add_validator_args(cls, parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default="validator",
    )

    parser.add_argument(
        "--neuron.model_id",
        type=str,
        help="The model to use for the validator.",
        default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    )

    parser.add_argument(
        "--neuron.tasks",
        type=str,
        nargs="+",
        help="The tasks to use for the validator.",
        default=list(TASKS.keys()),
    )
    import argparse

    def parse_probabilities(prob_list):
        try:
            # Convert each item in the list to a float
            return [float(p) for p in prob_list]
        except ValueError:
            raise argparse.ArgumentTypeError("All probabilities must be floats.")
        
    import argparse

    def parse_probabilities(prob_list):
        try:
            # Convert each item in the list to a float
            return [float(p) for p in prob_list]
        except ValueError:
            raise argparse.ArgumentTypeError("All probabilities must be floats.")
        
    parser.add_argument(
        "--neuron.task_p",
        type=parse_probabilities,  # Use the custom parsing function
        nargs="+",  # Allow multiple values
        type=parse_probabilities,  # Use the custom parsing function
        nargs="+",  # Allow multiple values
        help="The probability of sampling each task.",
        default=[1.0 / len(TASKS)] * len(TASKS),
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=15,
    )

    parser.add_argument(
        "--neuron.max_tokens",
        type=int,
        help="The maximum number of tokens in generated responses.",
        default=256,
    )

    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        help="The number of concurrent forwards running at any time.",
        default=1,
    )

    parser.add_argument(
        "--neuron.sample_size",
        type=int,
        help="The number of miners to query in a single step.",
        default=100,
    )

    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Disables setting weights.",
        default=False,
    )

    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        help="Moving average alpha parameter, how much to add of the new observation.",
        default=0.1,
    )

    parser.add_argument(
        "--neuron.decay_alpha",
        type=float,
        help="Constant decay rate for the moving average score.",
        default=0.001,
    )

    parser.add_argument(
        "--neuron.axon_off",
        "--axon_off",
        action="store_true",
        # Note: the validator needs to serve an Axon with their IP or they may
        #   be blacklisted by the firewall of serving peers on the network.
        help="Set this flag to not attempt to serve an Axon.",
        default=False,
    )

    parser.add_argument(
        "--neuron.vpermit_tao_limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=4096,
    )

    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="prompting-validators",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="macrocosmos",
    )

    parser.add_argument(
        "--neuron.query_unique_coldkeys",
        action="store_true",
        help="Only query a single hotkey per coldkey.",
        default=False,
    )

    parser.add_argument(
        "--neuron.query_unique_ips",
        action="store_true",
        help="Only query a single hotkey per ip.",
        default=False,
    )

    parser.add_argument(
        "--neuron.forward_max_time",
        type=int,
        help="Max time to wait for a forward call to complete in seconds.",
        default=120,
    )

    parser.add_argument(
        "--neuron.organic_sample_size",
        type=int,
        help="The number of miners to organic query in a single step.",
        default=5,
    )

    parser.add_argument(
        "--neuron.organic_sampling_mode",
        type=str,
        help="The mode for sampling miners using organic queries. Options include 'random' for random selection, "
            "'top_incentive' for selecting based on highest incentives.",
        default="random",
    )

    parser.add_argument(
        "--neuron.organic_disabled",
        action="store_true",
        help="Set this flag to disable organic scoring.",
        default=False,
    )

    # TODO: Set organic weight setting enabled by default after Aug 1, 2024.
    parser.add_argument(
        "--neuron.organic_set_weights_enabled",
        action="store_true",
        help="Set this flag to enable organic scoring weight setting.",
        default=False,
    )

    parser.add_argument(
        "--neuron.organic_synth_reward_scale",
        type=float,
        help="Scale factor for synthetic organic rewards.",
        default=0.1,
    )

    parser.add_argument(
        "--neuron.organic_reuse_response_disabled",
        action="store_true",
        help="If set, miner responses will be re-generated during reward generation. "
             "The default behavior is to reuse responses.",
        default=False,
    )

    parser.add_argument(
        "--neuron.organic_timeout",
        type=int,
        help="Organic query timeout for each call in seconds.",
        default=30,
    )

    parser.add_argument(
        "--neuron.organic_reference_max_tokens",
        type=int,
        help="Organic query timeout for each call in seconds.",
        default=1024,
    )

    # TODO: Increase sampling rate after after Aug 1, 2024.
    parser.add_argument(
        "--neuron.organic_trigger_frequency",
        type=float,
        help="Organic query sampling frequency (seconds or steps value).",
        default=120.0,
    )

    parser.add_argument(
        "--neuron.organic_trigger_frequency_min",
        type=float,
        help="Minimum organic query sampling frequency (seconds or steps value).",
        default=5.0,
    )

    parser.add_argument(
        "--neuron.organic_scaling_factor",
        type=float,
        help=(
            "The scaling factor to adjust the trigger frequency based on the size of the organic queue. "
            "A higher value means the trigger frequency adjusts more slowly to the increase of organic queue size."
        ),
        default=1.0,
    )

    parser.add_argument(
        "--neuron.organic_trigger",
        type=str,
        help="Organic query validation trigger mode (seconds or steps).",
        default="seconds",
    )

    parser.add_argument(
        "--neuron.organic_whitelist_hotkey",
        type=str,
        help="Allow request from specific hotkey. Defaults to OTF hotkey.",
        # OTF hotkey.
        default="5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3",
    )



def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)
