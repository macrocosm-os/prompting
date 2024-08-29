import json
import numpy as np
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal

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

    # wandb_config = {key: copy.deepcopy(self.config.get(key, None)) for key in ("neuron", "reward", "netuid", "wandb")}
    # wandb_config["neuron"].pop("full_path", None)
    wandb.login(anonymous="allow", key=settings.WANDB_API_KEY, verify=True)
    logger.info(f"Logging in to wandb on entity: {settings.WANDB_ENTITY} and project: {settings.WANDB_PROJECT_NAME}")
    WANDB = wandb.init(
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


class ErrorLoggingEvent(BaseModel):
    error: str
    forward_time: float | None = None


class ValidatorLoggingEvent(BaseModel):
    block: int
    step: int
    step_time: float
    response_event: DendriteResponseEvent
    forward_time: float | None = None
    task_id: str

    model_config = ConfigDict(arbitrary_types_allowed=True, copy_on_model_validation=False)

    def __str__(self):
        sample_completions = [completion for completion in self.response_event.completions if len(completion) > 0]
        sample_completion = sample_completions[0] if sample_completions else "All completions are empty"
        return f"""ValidatorLoggingEvent:
            Block: {self.block}
            Step: {self.step}
            Step Time: {self.step_time}
            forward_time: {self.forward_time}
            task_id: {self.task_id}
            Number of total completions: {len(self.response_event.completions)}
            Number of non-empty completions: {len(sample_completions)}
            Completions: {sample_completions}
            Sample completion: {sample_completion}"""


class RewardLoggingEvent(BaseModel):
    best: str
    reward_events: list[WeightedRewardEvent]
    penalty_events: list[WeightedRewardEvent]
    task_id: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self):
        rewards = [r.reward_event.rewards for r in self.reward_events]

        return f"""RewardLoggingEvent:
            Best: {self.best}
            Rewards:
                Rewards: {rewards}
                Min: {np.min(rewards)}
                Max: {np.max(rewards)}
                Average: {np.mean(rewards)}
            Penalty Events: {self.penalty_events}
            task_id: {self.task_id}"""


class MinerLoggingEvent(BaseModel):
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


def log_event(event: ValidatorLoggingEvent | MinerLoggingEvent | ErrorLoggingEvent):
    if not settings.LOGGING_DONT_SAVE_EVENTS:
        logger.info(f"{event}")

    if settings.WANDB_ON:
        wandb.log(event.model_dump())
