import json
import os
import wandb
import bittensor as bt
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List


@dataclass
class Log:
    validator_model_id: str
    challenge: str
    challenge_prompt: str
    reference: str
    miners_ids: List[str]
    responses: List[str]
    miners_time: List[float]
    challenge_time: float
    reference_time: float
    rewards: List[float]
    task: dict
    # extra_info: dict


# def get_extra_log_info(agent: HumanAgent, references: List[str]) -> dict:
#     extra_info = {
#         'challenge_length_chars': len(agent.challenge),
#         'challenge_length_words': len(agent.challenge.split()),
#         'reference_length_chars': [len(reference) for reference in references],
#         'reference_length_words': [len(reference.split()) for reference in references],
#     }

#     return extra_info


def export_logs(logs: List[Log]):
    bt.logging.info("📝 Exporting logs...")

    # Create logs folder if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Get the current date and time for logging purposes
    date_string = datetime.now().strftime("%Y-%m-%d_%H:%M")

    all_logs_dict = [asdict(log) for log in logs]

    for logs in all_logs_dict:
        task_dict = logs.pop('task')
        prefixed_task_dict = {f'task_{k}': v for k, v in task_dict.items()}
        logs.update(prefixed_task_dict)

    log_file = f"./logs/{date_string}_output.json"
    with open(log_file, 'w') as file:
        json.dump(all_logs_dict, file)

    return log_file


def init_wandb(config):
    return wandb.init(
        anonymous="allow",
        project=config.wandb.project_name,
        entity=config.wandb.entity,
        tags=config.wandb.tags,
    )