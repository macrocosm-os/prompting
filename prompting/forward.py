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
import time
import wandb
import torch
import random
import asyncio

import numpy as np
import pandas as pd
import bittensor as bt

from typing import List
from types import SimpleNamespace
from dataclasses import asdict
from prompting.agent import HumanAgent
from prompting.dendrite import DendriteResponseEvent
from prompting.conversation import create_task
from prompting.protocol import Prompting
from prompting.rewards import RewardPipeline, RewardResult, RewardEvent, RewardModelTypeEnum
from prompting.utils.uids import get_random_uids
from prompting.utils.logging import init_wandb

from transformers import pipeline


async def run_step(self, agent: HumanAgent, k: int, timeout: float, exclude: list = []):
    """Executes a single step of the agent, which consists of:
    - Getting a list of uids to query
    - Querying the network
    - Rewarding the network
    - Updating the scores
    - Logging the event

    Args:
        agent (HumanAgent): The agent to run the step for.
        k (int): The number of uids to query.
        timeout (float): The timeout for the queries.
        exclude (list, optional): The list of uids to exclude from the query. Defaults to [].
    """

    bt.logging.debug("run_step", agent.task.name)

    # Record event start time.
    start_time = time.time()
    # Get the list of uids to query for this step.
    uids = get_random_uids(self, k=k, exclude=exclude).to(self.device)

    axons = [self.metagraph.axons[uid] for uid in uids]
    # TODO: Should this be entire history?
    synapse = Prompting(roles=["user"], messages=[agent.challenge])

    # Make calls to the network with the prompt.
    responses: List[bt.Synapse] = await self.dendrite(
        axons=axons,
        synapse=synapse,
        timeout=timeout,
    )

    # Encapsulate the responses in a response event (dataclass)
    response_event = DendriteResponseEvent(responses, uids)

    # Reward the responses and get the reward result (dataclass)
    # This contains a list of RewardEvents but can be exported as a dict (column-wise) for logging etc
    reward_result = RewardResult(self.reward_pipeline, task=agent.task, response_event=response_event)

    # The original idea was that the agent is 'satisfied' when it gets a good enough response (e.g. reward critera is met, such as ROUGE>threshold)
    agent.update_progress(
        top_reward=reward_result.rewards.max(),
        top_response=response_event.completions[reward_result.rewards.argmax()]
    )

    self.update_scores(uids, reward_result.rewards)

    # Log the step event.
    event = {
        "block": self.block,
        "step_time": time.time() - start_time,
        **asdict(agent.task), # can include time to use tools, create query/references
        **asdict(reward_result), # can include fine-gained rewards as well as times
        **asdict(response_event) # can include times, status, and completions
    }

    bt.logging.debug(f"Step complete. Event:\n{event}")
    if not self.config.neuron.dont_save_events:
        logger.log("EVENTS", "events", **event)

    # Log the event to wandb.
    if not self.config.wandb.off:
        self.wandb.log(event)

    return event



async def forward(self):

    bt.logging.info(f"📋 Selecting task... from {self.config.neuron.tasks} with distribution {self.config.neuron.task_p}")
    # Create a specific task
    task_name = np.random.choice(self.config.neuron.tasks, p=np.array(self.config.neuron.task_p)/sum(self.config.neuron.task_p))
    bt.logging.info(f"📋 Creating {task_name} task... ")
    task = create_task(self.llm_pipeline, task_name)


    # Create random agent with task, topic, profile...
    bt.logging.info(f"🤖 Creating agent for {task_name} task... ")
    agent = HumanAgent(task=task, llm=self.llm_pipeline, begin_conversation=True)

    rounds = 0
    exclude_uids = []
    while not agent.finished:
        # when run_step is called, the agent updates its progress
        event = await run_step(self, agent, k=self.config.sample_size, timeout=self.config.timeout, exclude=exclude_uids)
        exclude_uids += event['uids']

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

    class MockLogger:
        def __init__(self):
            self.history = []

        def log(self, level, name, **kwargs):
            self.history.append(kwargs)

    logger = MockLogger()

    #### CONFIG ####
    config = SimpleNamespace(
        model_id = "HuggingFaceH4/zephyr-7b-beta",
        neuron = SimpleNamespace(
            tasks=list(tasks_sampling_distribution.keys()),
            task_p=list(tasks_sampling_distribution.values()),
            moving_average_alpha=0.1,
        ),
        mock=True,
        sample_size=10,
        timeout=15,
        device='cuda',
        max_turns=1,
        termination_probability=1,
        wandb=wandb_config
    )

    #### GLOBAL SELF / NEURON ####
    llm_pipeline = pipeline(
        "text-generation",
        model_id=config.model_id,
        #device_map="cuda:0",
        device_map="auto",

        model_kwargs={
            "torch_dtype": torch.float16,
            # NOTE: LINE BELLOW IS TEMPORARY SINCE WE ONLY HAVE ONE FUNCTIONING GPU FOR 2 DIFFERENT USERS, SHOULD NOT BE USED IF GPU IS AVAILABLE
            "load_in_4bit": True
        }
    )

    from neurons.validator import Validator
    
    mock_self = Validator(config)

    # Note: Self could be abstracted into neuron class
    # mock_self = SimpleNamespace(
    #     config=config,
    #     llm_pipeline=llm_pipeline,
    #     reward_pipeline=RewardPipeline(selected_tasks=config.tasks),
    #     dendrite=MockDendrite(),
    #     subtensor=MockSubtensor(),
    #     moving_averaged_scores=torch.zeros(1024).to(config.device),
    #     device=config.device,
    #     wandb=init_wandb(config)
    # )


    #### FLOW EXECUTION ####
    num_steps = 4
    for _ in range(num_steps):
        asyncio.run(forward(mock_self))

    mock_self.wandb.finish()
    pd.DataFrame(mock_self.mock_log).to_csv('mock_log.csv')

