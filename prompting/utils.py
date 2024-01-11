import json
import os
import wandb
import bittensor as bt
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List
import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI


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


def load_llm(model: str, **kwargs):     
    bt.logging.info(f"🤖 Loading LLM model {model}...")   
    if model == 'zephyr':
        llm = HuggingFacePipeline.from_model_id(
            model_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            device=0,  # replace with device_map="auto" to use the accelerate library.
            #device_map="cuda:0",
            pipeline_kwargs={"max_new_tokens": 256},
            model_kwargs={ "torch_dtype": torch.bfloat16 }
        )        
    elif model.startswith('gpt'):
        llm = ChatOpenAI(model_name=model, max_tokens=256, **kwargs)
    else:
        raise NotImplementedError(f"Model {model} not implemented") 

    bt.logging.success(f"🤖 Loaded LLM model {model}!")
    return llm


def get_model_name_from_llm(llm) -> str:
    # Handles different naming rules for langchain wrappers                
    if hasattr(llm, 'model_name') and llm.model_name is not None:            
        return llm.model_name # model_name: OpenAI format
    else:
        return llm.model_id # model_id: HuggingFace
    

def init_wandb(config):
    return wandb.init(
        anonymous="allow",
        project=config.wandb.project_name,
        entity=config.wandb.entity,
        tags=config.wandb.tags,
    )