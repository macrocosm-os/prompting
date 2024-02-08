# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

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
import sys
import numpy as np
import bittensor as bt

from typing import Dict, List, Awaitable
from prompting.agent import HumanAgent
from prompting.dendrite import DendriteResponseEvent
from prompting.conversation import create_task
from prompting.protocol import StreamPromptingSynapse
from prompting.rewards import RewardResult
from prompting.utils.uids import get_random_uids
from prompting.utils.logging import log_event

async def handle_response(responses: Dict[int, List[Awaitable]]) -> List[bt.Synapse]:
    synapses = []

    for uid, resp in responses.items():
        async for chunk in resp:
            print(f"\nchunk for resp: {chunk}", end="", flush=True) #TODO Might want to do something else here. 
            
        synapse = (
            chunk  # last object yielded is the synapse itself with completion filled
        )

        #Double check to make sure chunk is a synapse
        if isinstance(synapse, StreamPromptingSynapse): 
            synapses.append(synapse)
        else: 
            synapses.append(
                StreamPromptingSynapse(roles = ['user'], messages = ['failure'], completion = '')
            )
            bt.logging.debug(f"Synapse is not StreamingPromptingSynapse. Miner uid {uid} completion set to '' ")


    return synapses

async def run_step(
    self, agent: HumanAgent, k: int, timeout: float, exclude: list = None
):
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
    uids = get_random_uids(self, k=k, exclude=exclude or []).to(self.device)
    uids_cpu = uids.cpu().tolist()

    axons = [self.metagraph.axons[uid] for uid in uids]

    # Make calls to the network with the prompt.
    # Important: the responses are always in the same order as the uids. 
    responses: List[StreamPromptingSynapse] = await self.dendrite(
        axons=axons,
        synapse=StreamPromptingSynapse(roles=["user"], messages=[agent.challenge]),
        timeout=timeout,
        deserialize = False, #Important to be False! 
        streaming = True
    )

    responses = await handle_response(responses = dict(zip(uids_cpu, responses)))

    # Encapsulate the responses in a response event (dataclass)
    response_event = DendriteResponseEvent(responses = responses, uids = uids)

    bt.logging.info(f"Created DendriteResponseEvent:\n {response_event}")
    # Reward the responses and get the reward result (dataclass)
    # This contains a list of RewardEvents but can be exported as a dict (column-wise) for logging etc
    reward_result = RewardResult(
        self.reward_pipeline,
        agent=agent,
        response_event=response_event,
        device=self.device,
    )
    bt.logging.info(f"Created RewardResult:\n {reward_result}")

    # The original idea was that the agent is 'satisfied' when it gets a good enough response (e.g. reward critera is met, such as ROUGE>threshold)
    agent.update_progress(
        top_reward=reward_result.rewards.max(),
        top_response=response_event.completions[reward_result.rewards.argmax()],
    )

    self.update_scores(reward_result.rewards, uids)

    # Log the step event.
    event = {
        "block": self.block,
        "step_time": time.time() - start_time,
        **agent.__state_dict__(full=self.config.neuron.log_full),
        **reward_result.__state_dict__(full=self.config.neuron.log_full),
        **response_event.__state_dict__(),
    }

    log_event(self, event)

    return event


async def forward(self):
    bt.logging.info("🚀 Starting forward loop...")

    while True:
        bt.logging.info(
            f"📋 Selecting task... from {self.config.neuron.tasks} with distribution {self.config.neuron.task_p}"
        )
        # Create a specific task
        task_name = np.random.choice(
            self.config.neuron.tasks, p=self.config.neuron.task_p
        )
        bt.logging.info(f"📋 Creating {task_name} task... ")
        try:
            task = create_task(llm_pipeline=self.llm_pipeline, task_name=task_name)
            break
        except Exception as e:
            bt.logging.error(
                f"Failed to create {task_name} task. {sys.exc_info()}. Skipping to next task."
            )
            continue

    # Create random agent with task, topic, profile...
    bt.logging.info(f"🤖 Creating agent for {task_name} task... ")
    agent = HumanAgent(
        task=task, llm_pipeline=self.llm_pipeline, begin_conversation=True
    )

    rounds = 0
    exclude_uids = []
    while not agent.finished:
        # when run_step is called, the agent updates its progress
        event = await run_step(
            self,
            agent,
            k=self.config.neuron.sample_size,
            timeout=self.config.neuron.timeout,
            exclude=exclude_uids,
        )
        exclude_uids += event["uids"]
        task.complete = True

        rounds += 1

    del agent
    del task
