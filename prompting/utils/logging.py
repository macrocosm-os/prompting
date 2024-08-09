import json
import os
import wandb
from wandb.wandb_run import Run
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal
from loguru import logger
import prompting
from prompting.settings import settings
from prompting.tasks.task_registry import TaskRegistry
from prompting.rewards.reward import WeightedRewardEvent
from prompting.base.dendrite import DendriteResponseEvent
from pydantic import BaseModel, ConfigDict

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
    # extra_info: dict


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

    # wandb_config = {key: copy.deepcopy(self.config.get(key, None)) for key in ("neuron", "reward", "netuid", "wandb")}
    # wandb_config["neuron"].pop("full_path", None)
    wandb.login(anonymous="allow", key=settings.WANDB_API_KEY, verify=True)
    logger.info(
        f"Logging in to wandb with key: {settings.WANDB_API_KEY[:6]}... on entity: {settings.WANDB_ENTITY} and project: {settings.WANDB_PROJECT_NAME}"
    )
    WANDB = wandb.init(
        # anonymous="allow",
        reinit=reinit,
        project=settings.WANDB_PROJECT_NAME,
        entity=settings.WANDB_ENTITY,
        mode="offline" if settings.WANDB_OFFLINE else "online",
        dir=settings.SAVE_PATH,
        tags=tags,
        notes=settings.WANDB_NOTES,
    )
    logger.success(f"Started a new wandb run <blue> {WANDB.name} </blue>")


def reinit_wandb(self):
    """Reinitializes wandb, rolling over the run."""
    WANDB.finish()
    init_wandb(self, reinit=True)


class ErrorEvent(BaseModel):
    error: str
    forward_time: float | None = None


class ValidatorEvent(BaseModel):
    best: str
    block: int
    step: int
    step_time: float
    reward_events: list[WeightedRewardEvent]
    penalty_events: list[WeightedRewardEvent]
    response_event: DendriteResponseEvent
    forward_time: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MinerEvent(BaseModel):
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


def log_event(event: ValidatorEvent | MinerEvent | ErrorEvent):
    if not settings.LOGGING_DONT_SAVE_EVENTS:
        logger.info(f"{event}")

    if not settings.WANDB_ON:
        return

    # Log the event to wandb.
    wandb.log(event.model_dump())
