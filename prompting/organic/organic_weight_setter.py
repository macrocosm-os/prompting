import sys
import time
import bittensor as bt
from prompting.agent import HumanAgent
from prompting.base.neuron import BaseNeuron
from prompting.conversation import create_task
from prompting.dendrite import DendriteResponseEvent
from prompting.organic import organic_task
from prompting.protocol import StreamPromptingSynapse

import asyncio
from starlette.types import Send
from functools import partial

import bittensor as bt

from prompting.forward import handle_response, log_stream_results, query_miners
from prompting.protocol import StreamPromptingSynapse
from prompting.organic.organic_dataset import OrganicDataset
from prompting.rewards.reward import RewardResult
from prompting.utils.logging import log_event
from prompting.utils.uids import get_random_uids, get_uids


class OrganicWeightSetter:
    def __init__(self, validator: BaseNeuron, axon: bt.axon, loop: asyncio.AbstractEventLoop):
        self._val = validator
        self._loop = loop
        self._axon = axon
        self._organic_dataset = OrganicDataset()
    
    def start_task(self):
        self._axon.attach(
            forward_fn=self._handle_organic,
            blacklist_fn=None,
            priority_fn=None,
        )
        self.loop.create_task(self._weight_setter())

    async def _weight_setter(self):
        # TODO (dbobrenko): Get rid of HumanAgent dependency.
        while True:
            timer_start = time.perf_counter()
            task_name = organic_task.TASK_NAME
            try:
                task = create_task(
                    llm_pipeline=self.llm_pipeline,
                    translation_pipeline=self.translation_pipeline,
                    task_name=task_name,
                    create_reference=False,
                )
            except Exception as e:
                bt.logging.error(f"Failed to create {task_name} task. {sys.exc_info()}. Skipping to next task.")
                continue
            agent = HumanAgent(task=task, llm_pipeline=self.llm_pipeline, begin_conversation=True)
            # sample = self._organic_dataset.random()
            roles = task.roles
            messages = task.messages
            event = await self._run_step(
                self,
                agent,
                roles=roles,
                messages=messages,
                k=self.config.neuron.sample_size,
                timeout=self.config.neuron.timeout,
                exclude=None,
            )

            # Adds forward time to event and logs it to wandb
            event["forward_time"] = time.perf_counter() - timer_start
            log_event(self, event)

    async def run_step(
        self, agent: HumanAgent, roles: list[str], messages: list[str], k: int, timeout: float, exclude: list = None
    ):
        """Executes a single step of the agent, which consists of:
        - Getting a list of uids to query
        - Querying the network
        - Rewarding the network
        - Updating the scores
        - Logging the event

        Args:
            agent (HumanAgent): The agent to run the step for.
            roles (List[str]): The roles for the synapse.
            messages (List[str]): The messages for the synapse.
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
        # TODO: if organic and response is ready
        streams_responses = await query_miners(self, roles, messages, uids, timeout)

        # Prepare the task for handling stream responses
        stream_results_dict = dict(zip(uids_cpu, streams_responses))
        tokenizer = self.llm_pipeline.tokenizer
        handle_stream_responses_task = asyncio.create_task(handle_response(stream_results_dict, tokenizer))

        # if not agent.task.static_reference:
        #     reference_generation_task = generate_reference(agent)
        #     _, stream_results = await asyncio.gather(
        #         reference_generation_task, handle_stream_responses_task
        #     )
        # else:
        stream_results = await handle_stream_responses_task

        log_stream_results(stream_results)

        # TODO: Create separate thread for consuming organic prompts, and return reward.
        # Encapsulate the responses in a response event (dataclass)
        response_event = DendriteResponseEvent(stream_results=stream_results, uids=uids, timeout=timeout)

        bt.logging.info(f"Created DendriteResponseEvent:\n {response_event}")
        # Reward the responses and get the reward result (dataclass)
        # This contains a list of RewardEvents but can be exported as a dict (column-wise) for logging etc
        bt.logging.info(f"Response from miners: {stream_results}")
        reward_result = RewardResult(
            self.reward_pipeline,
            agent=agent,
            response_event=response_event,
            device=self.device,
        )
        bt.logging.info(f"Created RewardResult:\n {reward_result}")

        best_response = response_event.completions[reward_result.rewards.argmax()]

        # The original idea was that the agent is 'satisfied' when it gets a good enough response (e.g. reward critera is met, such as ROUGE>threshold)
        agent.update_progress(
            top_reward=reward_result.rewards.max(),
            top_response=best_response,
        )

        self.update_scores(reward_result.rewards, uids)
        
        # Log the step event.
        event = {
            "best": best_response,
            "block": self.block,
            "step": self.step,
            "step_time": time.time() - start_time,
            **agent.__state_dict__(full=self.config.neuron.log_full),
            **reward_result.__state_dict__(full=self.config.neuron.log_full),
            **response_event.__state_dict__(),
        }

        return event

    async def _collect(self):
        while True:
            try:
                if self.organic_scoring_tasks:
                    completed, _ = await asyncio.wait(
                        self.organic_scoring_tasks,
                        timeout=1,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in completed:
                        if task.exception():
                            bt.logging.error(f"Error encountered in {OrganicWeightSetter.__name__} task")
                        else:
                            success, data = task.result()
                            if not success:
                                continue
                            self.total_scores += data[0]
                    self.organic_scoring_tasks.difference_update(completed)
                else:
                    await asyncio.sleep(60)
            except Exception as e:
                bt.logging.error(f"Error encountered in {OrganicWeightSetter.__name__}: {e}")
                await asyncio.sleep(10)

    async def _handle_organic(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        bt.logging.info(f"Organic handle: {synapse}")

        uids = get_uids(self._val, sampling_mode="random", k=self._val.config.neuron.organic_size, exclude=[])
        token_streamer = partial(self._query_miner_uids, synapse, uids)
        streaming_response = synapse.create_streaming_response(token_streamer)
        self._organic_dataset.add({"synapse": synapse, "response": streaming_response, "uids": uids})
        return streaming_response

    async def _query_miner_uids(self, synapse, uids, send: Send):
        bt.logging.info(f"Sending {synapse} request to UIDs: {uids}")
        responses = await query_miners(self._val, synapse.roles, synapse.messages, uids, self._val.config.neuron.timeout)
        return await self._stream_miner_responses(responses, send)

    async def _stream_miner_responses(self, responses, send: Send):
        for resp in responses:
            async for chunk in resp:
                if isinstance(chunk, str):
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    bt.logging.info(f"Streamed text: {chunk}")
            await send({"type": "http.response.body", "body": b"", "more_body": False})
