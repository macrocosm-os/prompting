import sys
import time
import bittensor as bt
import asyncio
from functools import partial
from typing import Awaitable, AsyncIterator

from starlette.types import Send
import bittensor as bt

from prompting.agent import HumanAgent
from prompting.base.neuron import BaseNeuron
from prompting.dendrite import DendriteResponseEvent
from prompting.organic import organic_task
from prompting.protocol import StreamPromptingSynapse
from prompting.forward import handle_response, log_stream_results, query_miners
from prompting.protocol import StreamPromptingSynapse
from prompting.organic.organic_dataset import OrganicDataset
from prompting.rewards.reward import RewardResult
from prompting.utils.logging import log_event
from prompting.utils.uids import get_random_uids, get_uids


class OrganicWeightSetter:
    def __init__(self, validator: BaseNeuron, axon: bt.axon, loop: asyncio.AbstractEventLoop):
        """Runs the organic weight setter task in separate threads.
        
        Creates 3 threads:
        - Receiving organic requests through axon.
        - Streaming response completions back to the caller through axon.
        - Queue to incentivize miner's completions for organic or synthetic organic queries.

        Args:
            validator (BaseNeuron): The validator to use.
            axon (bt.axon): Served and started axon.
            loop (asyncio.AbstractEventLoop): The loop to use.
        """
        # TODO (dbobrenko): Decouple HumanAgent dependency.
        # TODO (dbobrenko): Decouple OrganicTask dependency.
        # TODO (dbobrenko): Decouple OrganicDataset dependency.
        # TODO (dbobrenko): Decouple Validator dependecies: llm_pipeline, etc.
        self._val = validator
        self._loop = loop
        self._axon = axon
        self._organic_dataset = OrganicDataset()

    def start_task(self):
        validator_uid = self._val.metagraph.hotkeys.index(self._val.wallet.hotkey.ss58_address)
        bt.logging.info(f"Serving validator IP of UID {validator_uid} to chain...")
        self._axon = bt.axon(wallet=self._val.wallet, config=self._val.config)
        self._axon.attach(
            forward_fn=self._handle_organic,
            blacklist_fn=None,
            priority_fn=None,
        )
        self._axon.serve(netuid=self._val.config.netuid, subtensor=self._val.subtensor)
        self._axon.start()
        # try:
        #     asyncio.run_coroutine_threadsafe(self._weight_setter(), self._loop)
        #     bt.logging.info("Weight setter task started successfully.")
        # except Exception as e:
        #     bt.logging.error(f"Failed to start weight setter task: {e}")

    # async def _weight_setter(self):
    #     bt.logging.info("Entered _weight_setter")
    #     while True:
    #         bt.logging.info("Weight setter loop iteration")
    #         await asyncio.sleep(1)  # Simplified for testing

    async def _weight_setter(self):
        while True:
            timer_start = time.perf_counter()
            task_name = organic_task.TASK_NAME
            try:
                task = organic_task.OrganicTask(
                    llm_pipeline=self._val.llm_pipeline,
                    context=self._organic_dataset.next(),
                    create_reference=True,
                )
            except Exception as e:
                bt.logging.error(f"Failed to create {task_name} task. {sys.exc_info()}.")
                await asyncio.sleep(1)
                continue
            agent = HumanAgent(task=task, llm_pipeline=self._val.llm_pipeline, begin_conversation=True)
            # sample = self._organic_dataset.random()
            roles = task.roles
            messages = task.messages
            event = await self._run_step(
                agent,
                roles=roles,
                messages=messages,
                k=self._val.config.neuron.sample_size,
                timeout=self._val.config.neuron.timeout,
                exclude=None,
            )

            # Adds forward time to event and logs it to wandb.
            event["forward_time"] = time.perf_counter() - timer_start
            log_event(self._val, event)
            await asyncio.sleep(1)

    async def _run_step(
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
        start_time = time.perf_counter()
        # Get the list of uids to query for this step.
        uids = get_random_uids(self._val, k=k, exclude=exclude or []).to(self._val.device)
        uids_cpu = uids.cpu().tolist()
        # TODO: if organic and response is ready
        streams_responses = await query_miners(self._val, roles, messages, uids, timeout)

        # Prepare the task for handling stream responses
        stream_results_dict = dict(zip(uids_cpu, streams_responses))
        tokenizer = self._val.llm_pipeline.tokenizer
        stream_results = await handle_response(stream_results_dict, tokenizer)

        log_stream_results(stream_results)

        # TODO: Create separate thread for consuming organic prompts, and return reward.
        # Encapsulate the responses in a response event (dataclass)
        response_event = DendriteResponseEvent(stream_results=stream_results, uids=uids, timeout=timeout)

        bt.logging.info(f"Created DendriteResponseEvent:\n {response_event}")
        # Reward the responses and get the reward result (dataclass)
        # This contains a list of RewardEvents but can be exported as a dict (column-wise) for logging etc
        bt.logging.info(f"Response from miners: {stream_results}")
        reward_result = RewardResult(
            self._val.reward_pipeline,
            agent=agent,
            response_event=response_event,
            device=self._val.device,
        )
        bt.logging.info(f"Created RewardResult:\n {reward_result}")

        best_response = response_event.completions[reward_result.rewards.argmax()]

        # The original idea was that the agent is 'satisfied' when it gets a good enough response
        # (e.g. reward critera is met, such as ROUGE>threshold)
        agent.update_progress(
            top_reward=reward_result.rewards.max(),
            top_response=best_response,
        )

        self._val.update_scores(reward_result.rewards, uids)
        
        # Log the step event.
        event = {
            "best": best_response,
            "block": self._val.block,
            "step": self._val.step,
            "step_time": time.perf_counter() - start_time,
            **agent.__state_dict__(full=self._val.config.neuron.log_full),
            **reward_result.__state_dict__(full=self._val.config.neuron.log_full),
            **response_event.__state_dict__(),
        }

        return event

    async def _handle_organic(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        async def token_streamer(send: Send) -> Awaitable[None]:
            return await self._query_miner_uids(synapse, uids, send)

        # token_streamer = partial(self._query_miner_uids, synapse, uids)

        bt.logging.info(f"Organic handle: {synapse}")
        uids = get_uids(self._val, sampling_mode="random", k=self._val.config.neuron.organic_size, exclude=[])
        streaming_response = synapse.create_streaming_response(token_streamer)
        self._organic_dataset.add({"synapse": synapse, "response": streaming_response, "uids": uids})
        return streaming_response

    async def _query_miner_uids(self, synapse: StreamPromptingSynapse, uids, send: Send):
        bt.logging.info(f"Sending {synapse} request to UIDs: {uids}")
        responses = await query_miners(self._val, synapse.roles, synapse.messages, uids, self._val.config.neuron.timeout)
        return await self._stream_miner_responses(responses, send)

    async def _stream_miner_responses(self, responses: AsyncIterator, send: Send):
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

    async def _handle_organic1(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        async def _forward(
            self,
            synapse: StreamPromptingSynapse,
            uids,
            init_time: float,
            timeout_threshold: float,
            send: Send,
        ):
            buffer = []
            accumulated_chunks = []
            accumulated_chunks_timings = []
            messages = []
            temp_completion = ""  # for wandb logging
            timeout_reached = False

            try:
                start_time = time.time()

                responses = await query_miners(
                    self._val,
                    synapse.roles,
                    synapse.messages,
                    uids,
                    self._val.config.neuron.timeout
                )
                # system_prompt_message = [{"role": "system", "content": self.system_prompt}]
                # synapse_messages = [{"role": role, "content": message}
                #                     for role, message in zip(synapse.roles, synapse.messages)]
                
                # messages = system_prompt_message + synapse_messages
                
                # stream_response = self.model.chat.completions.create(
                #     model=self.config.neuron.model_id,
                #     messages=messages,
                #     temperature=self.config.neuron.temperature,
                #     max_tokens=self.config.neuron.max_tokens,
                #     stream=True
                # )

                async for chunk in responses:
                    if isinstance(chunk, list):
                        concatenated_chunks = "".join(chunk)
                        await send(
                            {
                                "type": "http.response.body",
                                "body": concatenated_chunks.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                        bt.logging.info(f"Streamed text: {chunk}")
                    if isinstance(chunk, str):
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                        bt.logging.info(f"Streamed text: {chunk}")
                    if chunk is not None and isinstance(chunk, StreamPromptingSynapse):
                        if len(self.accumulated_chunks) == 0:
                            self.accumulated_chunks.append(chunk.completion)
                            self.accumulated_chunks_timings.append(time.time() - start_time)
                        
                        self.finish_reason = "completed"
                        self.sequence_number += 1
                        # Assuming the last chunk holds the last value yielded which should be a synapse with the completion filled
                        synapse = chunk 
                        
                        await send(
                            {
                                "type": "http.response.body",
                                "body": synapse.completion.encode("utf-8"),
                                "more_body": True,
                            }
                        )

            #         chunk_content = chunk.choices[0].delta.content
            #         if chunk_content is None:
            #             bt.logging.info("OpenAI returned chunk content with None")
            #             continue

            #         accumulated_chunks.append(chunk_content)
            #         accumulated_chunks_timings.append(time.time() - start_time)

            #         buffer.append(chunk_content)

            #         if time.time() - init_time > timeout_threshold:
            #             bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
            #             timeout_reached = True
            #             break

            #         if len(buffer) == self._val.config.neuron.streaming_batch_size:
            #             joined_buffer = "".join(buffer)
            #             temp_completion += joined_buffer
            #             bt.logging.debug(f"Streamed tokens: {joined_buffer}")

            #             await send(
            #                 {
            #                     "type": "http.response.body",
            #                     "body": joined_buffer.encode("utf-8"),
            #                     "more_body": True,
            #                 }
            #             )
            #             buffer = []

            #     if (
            #         buffer and not timeout_reached
            #     ):  # Don't send the last buffer of data if timeout.
            #         joined_buffer = "".join(buffer)
            #         await send(
            #             {
            #                 "type": "http.response.body",
            #                 "body": joined_buffer.encode("utf-8"),
            #                 "more_body": False,
            #             }
            #         )

            except Exception as e:
                bt.logging.error(f"Error in forward: {e}")
            #     # bt.logging.error(print_exception(type(e), e, e.__traceback__))
            #     if self._val.config.neuron.stop_on_forward_exception:
            #         self.should_exit = True

            finally:
                synapse_latency = time.time() - init_time
            #     # if self.config.wandb.on:
            #     #     self.log_event(
            #     #         synapse=synapse,
            #     #         timing=synapse_latency,
            #     #         messages=messages,
            #     #         accumulated_chunks=accumulated_chunks,
            #     #         accumulated_chunks_timings = accumulated_chunks_timings,
            #     #     )

        bt.logging.debug(f"📧 Message received from {synapse.dendrite.hotkey}, IP: {synapse.dendrite.ip}; \nForwarding synapse: {synapse}")

        init_time = time.time()
        timeout_threshold = synapse.timeout

        uids = get_uids(self._val, sampling_mode="random", k=self._val.config.neuron.organic_size, exclude=[])
        token_streamer = partial(
            _forward,
            self,
            synapse,
            uids,
            init_time,
            timeout_threshold,
        )

        response = synapse.create_streaming_response(token_streamer)

        self._organic_dataset.add({"synapse": synapse, "response": response, "uids": uids})
        return response
