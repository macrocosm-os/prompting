# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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
# DEALINGS IN
#  THE SOFTWARE.
import asyncio
import time
import torch
import random
import bittensor as bt
import random
import numpy as np
import pandas as pd
import wandb
from typing import List
from types import SimpleNamespace
from dataclasses import asdict
from agent import HumanAgent
from tasks import DebuggingTask, QuestionAnsweringTask, SummarizationTask, MathTask, DateQuestionAnsweringTask
from agent import HumanAgent
from tools import WikiDataset, CodingDataset, MathDataset, DateQADataset
from protocol import Prompting
from transformers import pipeline
from rewards import RewardPipeline, RewardEvent, RewardModelTypeEnum
from network import NetworkResponseEvent, ttl_get_block, update_ema_scores, MockDendrite
from utils import init_wandb


def get_rewards(self, task, rewards_events: List[RewardEvent]) -> torch.FloatTensor:
    # TODO: How would using the Agent as a reward model fit into this flow?
    # Compute the rewards for the responses given the prompt
    # Creates a dict with the uids as keys and the final rewards as values
    uids_final_rewards = {}

    for task_reward_definition in task.reward_definition:
        # Gets appropriate reward event for the reward model defined in the task
        reward_event = next((event for event in rewards_events if task_reward_definition['name'] == event.model), None)

        if reward_event.model_type == RewardModelTypeEnum.WEIGHTED_REWARD:
            for uid, reward in zip(reward_event.uids, reward_event.rewards):
                # Sets uid as int instead of tensor
                uid = uid.item()
                # Multiplies the reward by the weight defined in the task
                final_rewards = task_reward_definition['weight'] * reward
                # Adds the reward to the uid's final reward
                uid_reward = uids_final_rewards.get(uid, 0)
                uids_final_rewards[uid] = uid_reward + final_rewards

        elif reward_event.model_type == RewardModelTypeEnum.FILTER_REWARD:
            ...
        elif reward_event.model_type == RewardModelTypeEnum.PENALTY:
            ...
        else:
            raise ValueError(f'Reward model type {reward_event.model_type} not supported.')

    final_rewards = torch.tensor(list(uids_final_rewards.values())).to(self.device)

    return final_rewards



async def run_step(self, agent: HumanAgent, k: int, timeout: float, exclude: list = []):
    bt.logging.debug("run_step", agent.task.name)

    # Record event start time.
    start_time = time.time()
    # Get the list of uids to query for this step.
    # TODO: implement production mode of uids
    #uids = get_random_uids(self, k=k, exclude=exclude).to(self.device)
    uids = []

    #axons = [self.metagraph.axons[uid] for uid in uids]
    synapse = Prompting(roles=["user"], messages=[agent.challenge])

    # Make calls to the network with the prompt.
    responses: List[bt.Synapse] = await self.dendrite(
     #   axons=axons,
        synapse=synapse,
        timeout=timeout,
    )

    # TODO: Remove for original implementation
    uids = torch.randint(1, 1025, (len(responses),)).to(self.device)

    network_response_event = NetworkResponseEvent(agent.task.reference, responses, uids)

    # Process the responses and truncates/adjusts them if necessary.
    # If we create some dataclass with the task and responses t

    # List of RewardEvents, one for each reward model
    rewards_events: RewardEvent = self.reward_pipeline.reward_responses(agent.task, network_response_event)

    final_rewards = get_rewards(self, agent.task, rewards_events)


    # The original idea was that the agent is 'satisfied' when it gets a good enough response (e.g. reward critera is met, such as ROUGE>threshold)
    top_reward = max(final_rewards)
    top_response = network_response_event.completions[final_rewards.argmax()]
    agent.update_progress(top_reward, top_response)

    update_ema_scores(self, uids, final_rewards)

    reward_events_dict = [reward_event.asdict() for reward_event in rewards_events]
    rewards_log_aggregation = {**{key: value for dict in reward_events_dict for key, value in dict.items()}}

    # Log the step event.
    event = {
        "block": ttl_get_block(self),
        "step_length": time.time() - start_time,
        **rewards_log_aggregation, # can include fine-gained rewards as well as times
        **asdict(agent.task), # can include time to use tools, create query/references
        **network_response_event.as_dict() # can include times, status, and completions
    }

    # bt.logging.debug("event:", str(event))
    # if not self.config.neuron.dont_save_events:
    #     logger.log("EVENTS", "events", **event)

    # Log the event to wandb.
    if not self.config.wandb.off:
        # wandb_event = EventSchema.from_dict(
        #     event, self.config.neuron.disable_log_rewards
        # )
        self.wandb.log(event)

    # Return the event.
    return event




def create_task(llm_pipeline, task_name):
    wiki_based_tasks = ['summarization', 'qa']
    coding_based_tasks = ['debugging']
    #TODO Add math and date_qa to this structure

    # TODO: Abstract dataset classes into common dynamic interface
    if task_name in wiki_based_tasks:
        dataset = WikiDataset()

    elif task_name in coding_based_tasks:
        dataset = CodingDataset()

    elif task_name == 'math':
        dataset = MathDataset()

    elif task_name == 'date_qa':
        dataset = DateQADataset()


    if task_name == 'summarization':
        return SummarizationTask(llm_pipeline=llm_pipeline, context=dataset.next())

    elif task_name == 'qa':
        return QuestionAnsweringTask(llm_pipeline=llm_pipeline, context=dataset.next())

    elif task_name == 'debugging':
        return DebuggingTask(llm_pipeline=llm_pipeline, context=dataset.next())

    elif task_name == 'math':
        return MathTask(llm_pipeline=llm_pipeline, context=dataset.next())

    elif task_name == 'date_qa':
        return DateQuestionAnsweringTask(llm_pipeline=llm_pipeline, context=dataset.next())

    else:
        raise ValueError(f'Task {task_name} not supported. Please choose a valid task')


async def forward(self):
    # Create a specific task
    task_name = np.random.choice(self.config.tasks, p=self.config.task_distribution)
    bt.logging.info(f"📋 Creating {task_name} task... ")
    task = create_task(self.llm_pipeline, task_name)


    # Create random agent with task, topic, profile...
    bt.logging.info(f"🤖 Creating agent for {task_name} task... ")
    agent = HumanAgent(task=task, llm=self.llm_pipeline, begin_conversation=True)

    rounds = 0
    exclude_uids = []
    while not agent.finished:
        ## TODO: Add k and timeout neuron parameters
        # when run_step is called, the agent updates its progress
        event = await run_step(self, agent, k=10, timeout=15, exclude=exclude_uids)
        print(event)
        self.mock_log.append(event)

        #exclude_uids += event['uids']

        ## TODO: Add max_turns and termination_probability parameters
        if rounds > self.config.max_turns or random.random() < self.config.termination_probability:
            break

        rounds += 1




if __name__ == "__main__":
    # NOTE: TASKS MATH AND DATE_QA ARE NOT WORKING
    tasks_sampling_distribution = {
        'debugging':0.0,
        'qa': 0.0,
        'summarization': 0.0,
        'math': 1.0,
        'date_qa':0.0
    }

    # Filter out tasks with 0 probability of being sampled to be highlighted in wandb
    sampled_tasks = [key for key, value in tasks_sampling_distribution.items() if value != 0]
    wandb_config = SimpleNamespace(
        project_name="agent_experiments",
        entity="sn1",
        # NOTE: CHECK APPROPIATE TAGS FOR YOUR TEST RUN
        tags=['MOCK_TEST', 'zephyr_4bits'] + sampled_tasks,
        off=False,
    )

    #### CONFIG ####
    config = SimpleNamespace(
        tasks=list(tasks_sampling_distribution.keys()),
        task_distribution=list(tasks_sampling_distribution.values()),
        device='cuda',
        neuron= SimpleNamespace(
            moving_average_alpha=0.1,
        ),
        max_turns=1,
        termination_probability=1,
        wandb=wandb_config
    )

    #### GLOBAL SELF / NEURON ####
    llm_pipeline = pipeline(
        "text-generation",
        model="HuggingFaceH4/zephyr-7b-beta",
        #device_map="cuda:0",
        device_map="auto",

        model_kwargs={
            "torch_dtype": torch.float16,
            # NOTE: LINE BELLOW IS TEMPORARY SINCE WE ONLY HAVE ONE FUNCTIONING GPU FOR 2 DIFFERENT USERS, SHOULD NOT BE USED IF GPU IS AVAILABLE
            "load_in_4bit": True
        }
    )

    # Note: Self could be abstracted into neuron class
    mock_self = SimpleNamespace(
        config=config,
        llm_pipeline=llm_pipeline,
        reward_pipeline=RewardPipeline(selected_tasks=config.tasks),
        dendrite=MockDendrite(),
        moving_averaged_scores=torch.zeros(1024).to(config.device),
        device=config.device,
        mock_log=[],
        wandb=init_wandb(config)
    )


    #### FLOW EXECUTION ####
    num_steps = 4
    for _ in range(num_steps):
        asyncio.run(forward(mock_self))

    mock_self.wandb.finish()
    pd.DataFrame(mock_self.mock_log).to_csv('mock_log.csv')

