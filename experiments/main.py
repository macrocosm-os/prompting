import bittensor as bt
import time
from prompting.tasks import Task, SummarizationTask, WikiDataset, QuestionAnsweringTask
from typing import List
from experiments.miners import Miner, NetworkResponse, MockMiner
from prompting.utils import export_logs, Log
import openai
import os
from dotenv import load_dotenv, find_dotenv
from prompting.rewards.rouge import calculate_rouge_scores
from prompting.agent import HumanAgent
import random
import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI

class Neuron:
    def __init__(self, llm):
        self.llm = llm


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



def load_neuron(model: str) -> Neuron:        
    # Load OpenAI API key from .env file for test purposes, remove for production
    _ = load_dotenv(find_dotenv()) 
    openai.api_key = os.environ['OPENAI_API_KEY']
    
    # Loads LLM from model name
    llm = load_llm(model)
    neuron = Neuron(llm)    
    return neuron


def create_random_task(llm) -> Task:
    # TODO: Implement data seed creation
    # TODO: Implement random task creation    
    # TODO: Modify tasks on ./tasks/* to support langchain
    dataset = Dataset()
    dataset = WikiDataset()
    task = SummarizationTask(dataset)    
    task = QuestionAnsweringTask()    
    return task



def create_random_task(llm) -> Task:        
    return random.choice([
        SummarizationTask(WikiDataset()), 
        QuestionAnsweringTask()
    ])


def query_network(query: str, llm) -> List[NetworkResponse]:
    # TODO: replace to real querying of the network once it's ready
    # Simulates querying the miners for the given query using multiple miners
    bt.logging.debug(f"💬 Querying miners for query: {query}")

    # zephyr_miner = Miner(llm)
    gpt4_miner = Miner(load_llm("gpt-4"))
    mock_miner = MockMiner(default_response="Austin is the capital of Texas", miner_id="static_response")

    # miners = [zephyr_miner, gpt4_miner, mock_miner]
    miners = [gpt4_miner, mock_miner]


    #TODO: Make async if takes too long
    responses = []
    for miner in miners:    
        response = miner.query(query)
        responses.append(response)

    return responses


def reward_responses(responses: List[str], reference: str):
   # TODO: Add time scoring function
    rewards = calculate_rouge_scores(responses, reference)
    return rewards


def update_weights_on_chain(responses: List[bt.Synapse], rewards: List[float]):
    # TODO: Implement update weights on chain
    # Note: To be implemented next to production phase, no priority for initial push
    pass


def forward(neuron) -> Log: 
    bt.logging.info("📋 Creating random task... ")
    task = create_random_task(neuron.llm)
    bt.logging.info("✅ Created task: " + task.name)

    agent = HumanAgent(task=task, llm=neuron.llm, begin_conversation=False)

    t0 = time.time()
    agent.challenge = agent.create_challenge()        
    challenge_time = time.time() - t0 
    
    t0 = time.time()
    agent.reference = agent.create_reference()        
    reference_time = time.time() - t0

    bt.logging.info("🌐 Querying network...")
    network_responses = query_network(agent.challenge, neuron.llm)
    responses = [response.response for response in network_responses]
    miners_ids = [response.miner_id for response in network_responses]
    miners_time = [response.time for response in network_responses]

    bt.logging.info("💰 Rewarding responses...")
    rewards = reward_responses(responses, agent.reference)
    
    bt.logging.info("🔗 Updating weights on chain...")
    update_weights_on_chain(responses, rewards)

    log = Log(
        miners_ids=miners_ids,
        challenge_time=challenge_time,
        reference_time=reference_time,
        responses=responses,
        miners_time=miners_time,
        validator_model_id=get_model_name_from_llm(neuron.llm),
        challenge=agent.challenge,
        challenge_prompt=agent.system_prompt,
        reference=agent.reference,
        rewards=rewards,
        task=task.asdict(),
        # extra_info=get_extra_log_info(agent, references)
    )    

    return log


if __name__ == "__main__":
    bt.logging.set_debug(True)
    bt.logging.set_trace(True)

    bt.logging.info("🧠 Loading neuron...")
    neuron = load_neuron("gpt-4")

    # Run one step, set defined number of steps for now to facilitate testing
    steps_to_run = 1
    logs = []    
    for _ in range(steps_to_run):
        log = forward(neuron)
        logs.append(log)

    log_file = export_logs(logs)
    bt.logging.info(f"✅ Done! Exported logs to: {log_file}")