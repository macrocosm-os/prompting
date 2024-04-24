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
import asyncio
import numpy as np
import bittensor as bt
import traceback
from typing import List, Dict, Awaitable
from prompting.agent import HumanAgent
from prompting.dendrite import DendriteResponseEvent
from prompting.conversation import create_task
from prompting.protocol import StreamPromptingSynapse
from prompting.rewards import RewardResult
from prompting.utils.uids import get_random_uids
from prompting.utils.logging import log_event
from prompting.utils.misc import async_log, serialize_exception_to_string
from dataclasses import dataclass


@async_log
async def generate_reference(agent):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, agent.task.generate_reference, agent.llm_pipeline
    )
    return result


@async_log
async def execute_dendrite_call(dendrite_call):
    responses = await dendrite_call
    return responses


@dataclass
class StreamResult:
    synapse: StreamPromptingSynapse = None
    exception: BaseException = None
    uid: int = None


async def process_response(uid: int, async_generator: Awaitable):
    """Process a single response asynchronously."""
    try:
        chunk = None  # Initialize chunk with a default value
        async for (
            chunk
        ) in (
            async_generator
        ):  # most important loop, as this is where we acquire the final synapse.
            bt.logging.debug(f"\nchunk for uid {uid}: {chunk}")

        if chunk is not None:
            synapse = chunk  # last object yielded is the synapse itself with completion filled

            # Assuming chunk holds the last value yielded which should be a synapse
            if isinstance(synapse, StreamPromptingSynapse):
                return synapse

        bt.logging.debug(
            f"Synapse is not StreamPromptingSynapse. Miner uid {uid} completion set to '' "
        )
    except Exception as e:
        # bt.logging.error(f"Error in generating reference or handling responses: {e}", exc_info=True)
        traceback_details = traceback.format_exc()
        bt.logging.error(
            f"Error in generating reference or handling responses for uid {uid}: {e}\n{traceback_details}"
        )

        failed_synapse = StreamPromptingSynapse(
            roles=["user"], messages=["failure"], completion=""
        )

        return failed_synapse


@async_log
async def handle_response(responses: Dict[int, Awaitable]) -> List[StreamResult]:
    """The handle_response function is responsible for creating asyncio tasks around acquiring streamed miner chunks
    and processing them asynchronously. It then pairs the results with their original UIDs and returns a list of StreamResults.

    Args:
        responses (Dict[int, Awaitable]): Responses contains awaitables that are used to acquire streamed miner chunks.

    Raises:
        ValueError

    Returns:
        List[StreamResult]: DataClass containing the synapse, exception, and uid
    """
    tasks_with_uid = [
        (uid, responses[uid]) for uid, _ in responses.items()
    ]  # Pair UIDs with their tasks

    # Start tasks, preserving order and their associated UIDs
    tasks = [process_response(uid, resp) for uid, resp in tasks_with_uid]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    mapped_results = []
    # Pair each result with its original uid
    for (uid, _), result in zip(tasks_with_uid, results):
        # If the result is a StreamPromptingSynapse, the response was successful and the stream result is added without exceptions
        if isinstance(result, StreamPromptingSynapse):
            mapped_results.append(StreamResult(synapse=result, uid=uid))

        # If the result is an exception, the response was unsuccessful and the stream result is added with the exception and an empty synapse
        elif isinstance(result, BaseException):
            failed_synapse = StreamPromptingSynapse(
                roles=["user"], messages=["failure"], completion=""
            )
            mapped_results.append(
                StreamResult(synapse=failed_synapse, exception=result, uid=uid)
            )

        # If the result is neither an error or a StreamSynapse, log the error and raise a ValueError
        else:
            bt.logging.error(f"Unexpected result type for UID {uid}: {result}")
            raise ValueError(f"Unexpected result type for UID {uid}: {result}")

    return mapped_results


@async_log
async def generate_reference(agent: HumanAgent):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, agent.task.generate_reference, agent.llm_pipeline
    )
    return result


def log_stream_results(stream_results: List[StreamResult]):
    failed_responses = [
        response for response in stream_results if response.exception is not None
    ]
    empty_responses = [
        response
        for response in stream_results
        if response.exception is None and response.synapse.completion == ""
    ]
    non_empty_responses = [
        response
        for response in stream_results
        if response.exception is None and response.synapse.completion != ""
    ]

    bt.logging.info(f"Total of non_empty responses: ({len(non_empty_responses)})")
    bt.logging.info(f"Total of empty responses: ({len(empty_responses)})")
    bt.logging.info(
        f"Total of failed responses: ({len(failed_responses)}):\n {failed_responses}"
    )

    for failed_response in failed_responses:
        formatted_exception = serialize_exception_to_string(failed_response.exception)
        bt.logging.error(
            f"Failed response for uid {failed_response.uid}: {formatted_exception}"
        )


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

    # Directly call dendrite and process responses in parallel
    streams_responses = await self.dendrite(
        axons=axons,
        synapse=StreamPromptingSynapse(roles=["user"], messages=[agent.challenge]),
        timeout=timeout,
        deserialize=False,
        streaming=True,
    )

    # Prepare the task for handling stream responses
    handle_stream_responses_task = asyncio.create_task(
        handle_response(responses=dict(zip(uids_cpu, streams_responses)))
    )

    if not agent.task.static_reference:
        reference_generation_task = generate_reference(agent)
        _, stream_results = await asyncio.gather(
            reference_generation_task, handle_stream_responses_task
        )
    else:
        stream_results = await handle_stream_responses_task

    log_stream_results(stream_results)

    all_synapses_results = [stream_result.synapse for stream_result in stream_results]

    # Encapsulate the responses in a response event (dataclass)
    response_event = DendriteResponseEvent(
        responses=all_synapses_results, uids=uids, timeout=timeout
    )

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

    stream_results_uids = [stream_result.uid for stream_result in stream_results]
    stream_results_exceptions = [
        serialize_exception_to_string(stream_result.exception)
        for stream_result in stream_results
    ]
    # Log the step event.
    event = {
        "block": self.block,
        "step_time": time.time() - start_time,
        "stream_results_uids": stream_results_uids,
        "stream_results_exceptions": stream_results_exceptions,
        **agent.__state_dict__(full=self.config.neuron.log_full),
        **reward_result.__state_dict__(full=self.config.neuron.log_full),
        **response_event.__state_dict__(),
    }

    return event


async def forward(self):
    bt.logging.info("🚀 Starting forward loop...")
    forward_start_time = time.time()

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
            task = create_task(
                llm_pipeline=self.llm_pipeline,
                task_name=task_name,
                create_reference=False,
            )
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
        # Note: The try catch is a safe clause to ensure that the forward loop continues even if an error occurs in run_step.
        # To be reconsidered in the next version.
        try:
            # when run_step is called, the agent updates its progress
            event = await run_step(
                self,
                agent,
                k=self.config.neuron.sample_size,
                timeout=self.config.neuron.timeout,
                exclude=exclude_uids,
            )

            # Adds forward time to event and logs it to wandb
            event["forward_time"] = time.time() - forward_start_time
            log_event(self, event)

            exclude_uids += event["uids"]
            task.complete = True

            rounds += 1
        except BaseException as e:
            unexpected_errors = serialize_exception_to_string(e)
            bt.logging.error(
                f"Error in run_step: Skipping to next round. \n {unexpected_errors}"
            )

            event = {"unexpected_errors": unexpected_errors}

            log_event(self, event)
            continue

    del agent
    del task
