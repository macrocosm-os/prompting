import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import wandb
from loguru import logger
from pydantic import BaseModel, ConfigDict
from wandb.wandb_run import Run

import prompting
from prompting.rewards.reward import WeightedRewardEvent
from prompting.tasks.task_registry import TaskRegistry
from shared.dendrite import DendriteResponseEvent
from shared.settings import shared_settings

WANDB: Run


@dataclass
class Log:
    validator_model_id: str
    challenge: str
    challenge_prompt: str
    reference: str
    miners_ids: list[str]
    responses: list[str]
    miners_time: list[float]
    challenge_time: float
    reference_time: float
    rewards: list[float]
    task: dict


def export_logs(logs: list[Log]):
    logger.info("📝 Exporting logs...")

    # Create logs folder if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Get the current date and time for logging purposes
    date_string = datetime.now().strftime("%Y-%m-%d_%H:%M")

    all_logs_dict = [asdict(log) for log in logs]

    for logs in all_logs_dict:
        task_dict = logs.pop("task")
        prefixed_task_dict = {f"task_{k}": v for k, v in task_dict.items()}
        logs.update(prefixed_task_dict)

    log_file = f"./logs/{date_string}_output.json"
    with open(log_file, "w") as file:
        json.dump(all_logs_dict, file)

    return log_file


def should_reinit_wandb():
    """Checks if 24 hours have passed since the last wandb initialization."""
    # Get the start time from the wandb config
    if wandb.run is None:
        return False
    wandb_start_time = wandb.run.config.get("wandb_start_time", None)

    if wandb_start_time:
        # Convert the stored time (string) back to a datetime object
        wandb_start_time = datetime.strptime(wandb_start_time, "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()
        elapsed_time = current_time - wandb_start_time
        # Check if more than 24 hours have passed
        if elapsed_time > timedelta(hours=shared_settings.MAX_WANDB_DURATION):
            return True
    return False


def init_wandb(reinit=False, neuron: Literal["validator", "miner"] = "validator", custom_tags: list = []):
    """Starts a new wandb run."""
    global WANDB
    tags = [
        f"Wallet: {shared_settings.WALLET.hotkey.ss58_address}",
        f"Version: {prompting.__version__}",
        # str(prompting.__spec_version__),
        f"Netuid: {shared_settings.NETUID}",
    ]

    if shared_settings.MOCK:
        tags.append("Mock")
    if shared_settings.NEURON_DISABLE_SET_WEIGHTS:
        tags.append("disable_set_weights")
        tags += [
            f"Neuron UID: {shared_settings.METAGRAPH.hotkeys.index(shared_settings.WALLET.hotkey.ss58_address)}",
            f"Time: {datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        ]

    tags += custom_tags

    task_list = []
    for task_config in TaskRegistry.task_configs:
        task_list.append(task_config.task.__name__)

    wandb_config = {
        "HOTKEY_SS58": shared_settings.WALLET.hotkey.ss58_address,
        "NETUID": shared_settings.NETUID,
        "wandb_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "TASKS": task_list,
    }
    wandb.login(anonymous="allow", key=shared_settings.WANDB_API_KEY, verify=True)
    logger.info(
        f"Logging in to wandb on entity: {shared_settings.WANDB_ENTITY} and project: {shared_settings.WANDB_PROJECT_NAME}"
    )
    WANDB = wandb.init(
        reinit=reinit,
        project=shared_settings.WANDB_PROJECT_NAME,
        entity=shared_settings.WANDB_ENTITY,
        mode="offline" if shared_settings.WANDB_OFFLINE else "online",
        dir=shared_settings.SAVE_PATH,
        tags=tags,
        notes=shared_settings.WANDB_NOTES,
        config=wandb_config,
    )
    signature = shared_settings.WALLET.hotkey.sign(WANDB.id.encode()).hex()
    wandb_config["SIGNATURE"] = signature
    WANDB.config.update(wandb_config)
    logger.success(f"Started a new wandb run <blue> {WANDB.name} </blue>")


def reinit_wandb():
    """Reinitializes wandb, rolling over the run."""
    global WANDB
    WANDB.finish()
    init_wandb(reinit=True)


class BaseEvent(BaseModel):
    forward_time: float | None = None


class WeightSetEvent(BaseEvent):
    weight_set_event: list[float]


class ErrorLoggingEvent(BaseEvent):
    error: str
    forward_time: float | None = None


class ValidatorLoggingEvent(BaseEvent):
    block: int
    step: int
    step_time: float
    response_event: DendriteResponseEvent
    task_id: str
    forward_time: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, copy_on_model_validation=False)

    def __str__(self):
        sample_completions = [completion for completion in self.response_event.completions if len(completion) > 0]
        return f"""ValidatorLoggingEvent:
            Block: {self.block}
            Step: {self.step}
            Step Time: {self.step_time}
            forward_time: {self.forward_time}
            task_id: {self.task_id}
            Number of total completions: {len(self.response_event.completions)}
            Number of non-empty completions: {len(sample_completions)}
            Sample Completions: {sample_completions[:5]}...
            """


class RewardLoggingEvent(BaseEvent):
    block: int | None
    step: int
    response_event: DendriteResponseEvent
    reward_events: list[WeightedRewardEvent]
    task_id: str
    reference: str
    challenge: str | list[dict]
    task: str
    task_dict: dict
    source: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self):
        rewards = [r.rewards for r in self.reward_events]

        return f"""RewardLoggingEvent:
            Rewards:
                Rewards: {rewards}
                Min: {np.min(rewards) if len(rewards) > 0 else None}
                Max: {np.max(rewards) if len(rewards) > 0 else None}
                Average: {np.mean(rewards) if len(rewards) > 0 else None}
            task_id: {self.task_id}
            task_name: {self.task}"""


class MinerLoggingEvent(BaseEvent):
    epoch_time: float
    messages: int
    accumulated_chunks: int
    accumulated_chunks_timings: float
    validator_uid: int
    validator_ip: str
    validator_coldkey: str
    validator_hotkey: str
    validator_stake: float
    validator_trust: float
    validator_incentive: float
    validator_consensus: float
    validator_dividends: float
    model_config = ConfigDict(arbitrary_types_allowed=True)


def log_event(event: BaseEvent):
    if not shared_settings.LOGGING_DONT_SAVE_EVENTS:
        logger.info(f"{event}")

    if shared_settings.WANDB_ON:
        if should_reinit_wandb():
            reinit_wandb()
        unpacked_event = unpack_events(event)
        unpacked_event = convert_arrays_to_lists(unpacked_event)
        logger.debug(f"""LOGGING WANDB EVENT: {unpacked_event}""")
        wandb.log(unpacked_event)


def unpack_events(event: BaseEvent) -> dict[str, Any]:
    """reward_events and penalty_events are unpacked into a list of dictionaries."""
    event_dict = event.model_dump()
    for key in list(event_dict.keys()):
        if key == "response_event":
            nested_dict = event_dict.pop(key)
            if isinstance(nested_dict, dict):
                event_dict.update(nested_dict)
    return event_dict


def convert_arrays_to_lists(data: dict) -> dict:
    return {key: value.tolist() if hasattr(value, "tolist") else value for key, value in data.items()}


if shared_settings.WANDB_ON and not shared_settings.MOCK:
    init_wandb()
