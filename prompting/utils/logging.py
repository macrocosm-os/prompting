import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal, Optional

import wandb
from loguru import logger
from pydantic import BaseModel, ConfigDict
from wandb.wandb_run import Run

import prompting
from prompting.base.dendrite import DendriteResponseEvent
from prompting.rewards.reward import WeightedRewardEvent
from prompting.settings import settings
from prompting.tasks.task_registry import TaskRegistry

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


def should_reinit_wandb(step: int):
    # Check if wandb run needs to be rolled over.
    return settings.WANDB_ON and step and step % settings.WANDB_RUN_STEP_LENGTH == 0


def init_wandb(reinit=False, neuron: Literal["validator", "miner"] = "validator", custom_tags: list = []):
    """Starts a new wandb run."""
    global WANDB
    tags = [
        f"Wallet: {settings.WALLET.hotkey.ss58_address}",
        f"Version: {prompting.__version__}",
        # str(prompting.__spec_version__),
        f"Netuid: {settings.NETUID}",
    ]

    if settings.MOCK:
        tags.append("Mock")
    for task_config in TaskRegistry.task_configs:
        tags.append(task_config.task.__name__)
    if settings.NEURON_DISABLE_SET_WEIGHTS:
        tags.append("disable_set_weights")
        tags += [
            f"Neuron UID: {settings.METAGRAPH.hotkeys.index(settings.WALLET.hotkey.ss58_address)}",
            f"Time: {datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        ]

    tags += custom_tags

    wandb_config = {
        "HOTKEY_SS58": settings.WALLET.hotkey.ss58_address,
        "NETUID": settings.NETUID,
    }
    wandb.login(anonymous="allow", key=settings.WANDB_API_KEY, verify=True)
    logger.info(
        f"Logging in to wandb on entity: {settings.WANDB_ENTITY} and project: {settings.WANDB_PROJECT_NAME}"
    )
    WANDB = wandb.init(
        reinit=reinit,
        project=settings.WANDB_PROJECT_NAME,
        entity=settings.WANDB_ENTITY,
        mode="offline" if settings.WANDB_OFFLINE else "online",
        dir=settings.SAVE_PATH,
        tags=tags,
        notes=settings.WANDB_NOTES,
        config=wandb_config
    )
    signature = settings.WALLET.hotkey.sign(WANDB.id.encode()).hex()
    wandb_config["SIGNATURE"] = signature
    WANDB.config.update(wandb_config)
    logger.success(f"Started a new wandb run <blue> {WANDB.name} </blue>")


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    WANDB.finish()
    init_wandb(self, reinit=True)


class BaseEvent(BaseModel):
    forward_time: float | None = None


class ErrorEvent(BaseEvent):
    error: str


class ValidatorEvent(BaseEvent):
    best: str
    block: int
    step: int
    step_time: float
    reward_events: list[WeightedRewardEvent]
    penalty_events: list[WeightedRewardEvent]
    response_event: DendriteResponseEvent
    reference: str
    challenge: str
    task: str
    rewards: list[float]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ValidatorOrganicEvent(ValidatorEvent):
    organic_turn: Optional[int]
    organic_time_sample: Optional[float]
    organic_time_responses: Optional[float]
    organic_time_rewards: Optional[float]
    organic_time_weights: Optional[float]
    organic_queue_size: Optional[int]


class MinerEvent(BaseEvent):
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
    if not settings.LOGGING_DONT_SAVE_EVENTS:
        logger.info(f"{event}")

    if settings.WANDB_ON:
        unpacked_event = unpack_events(event)
        unpacked_event = convert_arrays_to_lists(unpacked_event)
        wandb.log(unpacked_event)


def unpack_events(event: BaseEvent) -> dict[str, Any]:
    """reward_events and penalty_events are unpacked into a list of dictionaries."""
    event_dict = event.model_dump()
    for key in list(event_dict.keys()):
        if key.endswith("_events"):
            event_dict.update(extract_reward_event(event_dict.pop(key)))
        if key == "response_event":
            nested_dict = event_dict.pop(key)
            if isinstance(nested_dict, dict):
                event_dict.update(nested_dict)
    return event_dict


def extract_reward_event(reward_event: list[dict[str, Any]]) -> dict[str, Any]:
    flattened_reward_dict = {}
    for element in reward_event:
        name = element["reward_event"].pop("reward_model_name")
        element["reward_event"]["weight"] = element.pop("weight")
        reward_event = element.pop("reward_event")
        new_reward_event = {f"{name}_{key}": value for key, value in reward_event.items()}
        flattened_reward_dict.update(new_reward_event)
    return flattened_reward_dict

def convert_arrays_to_lists(data: dict) -> dict:
    return {key: value.tolist() if hasattr(value, "tolist") else value for key, value in data.items()}
