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

from typing import List
from prompting.agent import HumanAgent
from prompting.dendrite import DendriteResponseEvent
from prompting.conversation import TransitionMatrix, ContextChain
from prompting.persona import create_persona
from prompting.protocol import PromptingSynapse
from prompting.rewards import RewardResult
from prompting.tasks import TASKS
from prompting.utils.uids import get_random_uids
from prompting.utils.logging import log_event


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

    axons = [self.metagraph.axons[uid] for uid in uids]
    # Make calls to the network with the prompt.
    responses: List[PromptingSynapse] = await self.dendrite(
        axons=axons,
        synapse=PromptingSynapse(roles=["user"], messages=[agent.challenge]),
        timeout=timeout,
    )

    # Encapsulate the responses in a response event (dataclass)
    response_event = DendriteResponseEvent(responses, uids)

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

    # NOTE: begin_probs only defines the start tasks. If we want to completely disable certain tasks we need to that in probs
    mat = TransitionMatrix(
        labels=list(self.transition_probs.keys()),
        probs=list(self.transition_probs.values()),
        begin_probs=self.config.neuron.task_p
    )

    exclude_uids = []
    # create a persona that will persist through the conversation
    persona = create_persona()
    num_steps = np.random.randint(1, 10)
    chain = ContextChain(matrix=mat, num_steps=num_steps, seed=None, mock=False)

    bt.logging.info(f'Starting conversation with {num_steps} steps')
    for context in chain:
        task_name = chain.task_name
        task = TASKS[task_name](llm_pipeline=self.llm_pipeline, context=context)
        bt.logging.info(f"📋 Selected task: {task}")

        # Create an agent with the selected task and persona, and begin the conversation.
        bt.logging.info(f"🤖 Creating agent for {task_name} task... ")
        agent = HumanAgent(
            task=task, llm_pipeline=self.llm_pipeline, begin_conversation=True, persona=persona
        )

        # Perform a single step of the agent, consisting of querying, rewarding, updating scores and logging the event.
        event = await run_step(
            self,
            agent,
            k=self.config.neuron.sample_size,
            timeout=self.config.neuron.timeout,
            exclude=exclude_uids,
        )
        exclude_uids += event["uids"]

        del agent
        del task
