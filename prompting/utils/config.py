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
from loguru import logger


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

    log_level_exists = "EVENTS" in logger._core.levels
    if not config.neuron.dont_save_events and not log_level_exists:
        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            os.path.join(config.neuron.full_path, "events.log"),
            rotation=config.neuron.events_retention_size,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            level="EVENTS",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )


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
        default="NousResearch/Nous-Hermes-2-SOLAR-10.7B",
    )

    parser.add_argument(
        "--neuron.tasks",
        type=str,
        nargs="+",
        help="The tasks to use for the validator.",
        default=["summarization", "qa", "debugging", "math", "date_qa"],
    )

    parser.add_argument(
        "--neuron.task_p",
        type=float,
        nargs="+",
        help="The probability of sampling each task.",
        default=[0.25, 0.25, 0.0, 0.25, 0.25],
    )

    parser.add_argument(
        "--neuron.timeout",
        type=float,
        help="The timeout for each forward call in seconds.",
        default=10,
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
        default=50,
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
        default="alpha-validators",
    )

    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="The name of the project where you are sending the new run.",
        default="opentensor-dev",
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
