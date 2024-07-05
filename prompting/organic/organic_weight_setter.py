import sys
import threading
import time
import bittensor as bt
import asyncio
from functools import partial
from typing import Awaitable, AsyncIterator
from concurrent.futures import TimeoutError

from starlette.types import Send
import bittensor as bt
import torch

from prompting.agent import HumanAgent
from prompting.base.neuron import BaseNeuron
from prompting.dendrite import DendriteResponseEvent
from prompting.organic import organic_task
from prompting.protocol import StreamPromptingSynapse
from prompting.forward import QueryMinersManager, handle_response, log_stream_results, query_miners
from prompting.organic.organic_dataset import OrganicDataset
from prompting.rewards.reward import RewardResult
from prompting.utils.logging import log_event
from prompting.utils.uids import get_random_uids, get_uids


class OrganicWeightSetter:
    def __init__(self, validator: BaseNeuron, axon: bt.axon):
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
        self._axon = axon
        self.should_exit = False
        self.is_running = False
        self._organic_dataset = OrganicDataset()

    def start_task(self):

        try:
            # asyncio.run_coroutine_threadsafe(self._weight_setter(), self._loop)
            self.run_in_background_thread()
            bt.logging.info("Weight setter task started successfully.")
        except Exception as e:
            bt.logging.error(f"Failed to start weight setter task: {e}")

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting organic tasks in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self._weight_setter, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping organic tasks in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    async def simple_coroutine(self):
        bt.logging.debug("Entered simple_coroutine")
        await asyncio.sleep(1)
        return "simple result"

    def _weight_setter(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        ###
        self._axon = bt.axon(wallet=self._val.wallet, config=self._val.config)
        self._axon.attach(
            forward_fn=self._handle_organic,
            blacklist_fn=None,
            priority_fn=None,
        )
        self._axon.serve(netuid=self._val.config.netuid, subtensor=self._val.subtensor)
        self._axon.start()

        async def stop_event_loop():
            while not self.should_exit:
                await asyncio.sleep(0.1)
            loop.stop()
            bt.logging.debug("Event loop stopped.")

        loop.create_task(stop_event_loop())

        async def run_tasks_forever():
            while not self.should_exit:
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
                    continue
                agent = HumanAgent(task=task, llm_pipeline=self._val.llm_pipeline, begin_conversation=True)
                roles = task.roles
                messages = task.messages
                try:
                    # Schedule the async task in the new event loop
                    # event = await self.simple_coroutine()
                    event = await asyncio.wait_for(
                        self._run_step(
                            agent=agent,
                            roles=roles,
                            messages=messages,
                            k=self._val.config.neuron.organic_size,
                            timeout=self._val.config.neuron.timeout,
                            exclude=None,
                        ),
                        timeout=30
                    )

                    # Adds forward time to event and logs it to wandb.
                    event["forward_time"] = time.perf_counter() - timer_start
                    log_event(self._val, event)
                except asyncio.TimeoutError:
                    bt.logging.error(f"Failed to run {task_name} task. {sys.exc_info()}.")
                except TimeoutError:
                    bt.logging.error("Task timed out.")

        try:
            loop.run_until_complete(run_tasks_forever())
        finally:
            loop.close()


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
        # streams_responses = await query_miners(self._val, roles, messages, uids, timeout)
        bt.logging.info(f"Querying miners with organic reward prompts: {uids_cpu}")
        # streams_responses = await QueryMinersManager(validator=self._val).query_miners(
        streams_responses = await query_miners(
            self._val,
            roles,
            messages,
            uids,
            self._val.config.neuron.timeout
        )

        # Prepare the task for handling stream responses
        stream_results_dict = dict(zip(uids_cpu, streams_responses))
        tokenizer = self._val.llm_pipeline.tokenizer
        stream_results = await handle_response(stream_results_dict, tokenizer)

        log_stream_results(stream_results)

        # Encapsulate the responses in a response event (dataclass)
        response_event = DendriteResponseEvent(stream_results=stream_results, uids=uids, timeout=timeout)

        bt.logging.info(f"Created organic reward DendriteResponseEvent:\n {response_event}")
        # Reward the responses and get the reward result (dataclass)
        # This contains a list of RewardEvents but can be exported as a dict (column-wise) for logging etc
        bt.logging.info(f"Organic reward response from miners: {stream_results}")
        reward_result = RewardResult(
            self._val.reward_pipeline,
            agent=agent,
            response_event=response_event,
            device=self._val.device,
        )
        bt.logging.info(f"Created organic reward RewardResult:\n {reward_result}")

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
        from transformers import AutoTokenizer
        model_name = "casperhansen/llama-3-8b-instruct-awq"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        async def _forward(
            self,
            synapse: StreamPromptingSynapse,
            uids,
            init_time: float,
            timeout_threshold: float,
            send: Send,
        ):
            accumulated_chunks = []
            # accumulated_chunks_timings = []
            # messages = []
            # temp_completion = ""  # for wandb logging
            # timeout_reached = False
            try:
                # timer_start = time.perf_counter()
                bt.logging.info(f"Querying miners with organic prompts: {uids}")
                # uids_cpu = uids.cpu().tolist()

                responses = self._val.dendrite.query(
                    axons=[self._val.metagraph.axons[uid] for uid in uids],
                    # synapse=StreamPromptingSynapse(roles=synapse.roles, messages=synapse.messages),
                    synapse=synapse,
                    timeout=30,
                    deserialize=False,
                    streaming=False,
                    # Doesn't work
                    # streaming=True,
                )
                # responses = await query_miners(
                #     self._val,
                #     synapse.roles,
                #     synapse.messages,
                #     torch.tensor(uids).to(self._val.device),
                #     self._val.config.neuron.timeout
                # )

                # Prepare the task for handling stream responses
                stream_results_dict = dict(zip(uids, responses))
                # tokenizer = self._val.llm_pipeline.tokenizer
                # stream_results = await handle_response(stream_results_dict, tokenizer)
                # bt.logging.info(f"ORGANIC Response:\n {stream_results[0].synapse.completion}")
                # return await send(
                #     {
                #         "type": "http.response.body",
                #         "body": stream_results[0].synapse.completion.encode("utf-8"),
                #         "more_body": True,
                #     }
                # )

                bt.logging.info(f"Awaiting miners with organic prompts: {uids}")
                for chunks in responses:
                    # await asyncio.sleep(1)
                    async for chunk in chunks:
                        if isinstance(chunk, str):
                            accumulated_chunks.append(chunk)
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": chunk.encode("utf-8"),
                                    "more_body": True,
                                }
                            )
                            bt.logging.info(f"Streamed text: {chunk}")
                        # if chunk is not None and isinstance(chunk, StreamPromptingSynapse):
                        #     accumulated_chunks.append(chunk.completion)
                        #     # self.finish_reason = "completed"
                        #     # self.sequence_number += 1
                        #     # Assuming the last chunk holds the last value yielded which should be a synapse with the completion filled
                        #     synapse = chunk
                            
                        #     if len(accumulated_chunks) == 0:
                        #         await send(
                        #             {
                        #                 "type": "http.response.body",
                        #                 "body": synapse.completion.encode("utf-8"),
                        #                 "more_body": False,
                        #             }
                        #         )

            #         accumulated_chunks.append(chunk_content)
            #         accumulated_chunks_timings.append(time.perf_counter() - timer_start)
            #         buffer.append(chunk_content)
            #         if time.time() - init_time > timeout_threshold:
            #             bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
            #             timeout_reached = True
            #             break
            #     if (buffer and not timeout_reached):  # Don't send the last buffer of data if timeout.
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
                print(f"Organic response: {synapse}")
                print("".join(accumulated_chunks))
                # if self.config.wandb.on:
                #     self.log_event(
                #         synapse=synapse,
                #         timing=synapse_latency,
                #         messages=messages,
                #         accumulated_chunks=accumulated_chunks,
                #         accumulated_chunks_timings = accumulated_chunks_timings,
                #     )

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

        # response = synapse.create_streaming_response(token_streamer)
        # self._organic_dataset.add({"synapse": synapse, "response": response, "uids": uids})
        return synapse.create_streaming_response(token_streamer)
