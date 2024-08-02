import asyncio
import json
import time
from functools import partial
from typing import Any, AsyncGenerator, Literal, Sequence, Tuple, Union

import bittensor as bt
from prompting import settings
from organic_scoring import OrganicScoringBase
from organic_scoring.synth_dataset import SynthDatasetBase, SynthDatasetEntry
from starlette.types import Send
from typing_extensions import override
from bittensor.dendrite import dendrite

from prompting.base.dendrite import SynapseStreamResult
from neurons.forward import handle_response
from prompting.llms.vllm_llm import vLLMPipeline
from prompting.organic.organic_task import OrganicTask, OrganicRewardConfig
from prompting.base.protocol import StreamPromptingSynapse

# from prompting.rewards.reward import RewardResult

# from prompting.tasks.task import make_system_prompt

from transformers import PreTrainedTokenizerFast
from pydantic import BaseModel


class OrganicScoringPrompting(OrganicScoringBase, BaseModel):
    axon: bt.axon
    synth_dataset: Union[SynthDatasetBase, Sequence[SynthDatasetBase]]
    trigger_frequency: Union[float, int]
    llm_pipeline: vLLMPipeline
    dendrite: bt.dendrite
    metagraph: bt.metagraph
    update_scores: callable
    dendrite: bt.dendrite
    tokenizer: PreTrainedTokenizerFast
    metagraph: bt.metagraph
    get_random_uids: callable
    wallet: bt.wallet

    trigger_frequency_min: Union[float, int] = 5
    trigger_scaling_factor: Union[float, int] = 5
    trigger: Literal["seconds", "steps"]
    #     """Organic Scoring implementation.

    #     Organic scoring runs in a separate `asyncio` task and is triggered by a timer or a step counter.

    #     Process Workflow:
    #     1. Trigger Check: Upon triggering the rewarding process, the system checks if the organic queue is empty.
    #         If the queue is empty, synthetic datasets are used to bootstrap the organic scoring mechanism.
    #         Otherwise, samples from the organic queue are utilized.
    #     2. Data Processing: The sampled data is concurrently passed to the `_query_miners` and `_generate_reference`
    #         methods.
    #     3. Reward Generation: After receiving responses from miners and any reference data, the information
    #         is processed by the `_generate_rewards` method.
    #     4. Weight Setting: The generated rewards are then applied through the `_set_weights` method.
    #     5. Logging: Finally, the results can be logged using the `_log_results` method, along with all relevant data
    #         provided as arguments, and default time elapsed on each step of rewarding process.
    #     """

    # @override
    # async def start_loop(self, llm_pipeline: vLLMPipeline, dendrite: bt.dendrite, update_scores: callable):
    #     """The main loop for running the organic scoring task, either based on a time interval or steps."""
    #     while not self._should_exit:
    #         if self._trigger == "steps":
    #             while self._step_counter < self._trigger_frequency:
    #                 await asyncio.sleep(0.1)

    #         try:
    #             logs = await self.loop_iteration(
    #                 llm_pipeline=llm_pipeline, dendrite=dendrite, update_scores=update_scores
    #             )
    #             await self.wait_until_next(timer_elapsed=logs.get("organic_time_total", 0))
    #         except Exception as e:
    #             bt.logging.error(f"Error occured during organic scoring iteration:\n{e}")
    #             await asyncio.sleep(1)

    # @override
    # async def loop_iteration(
    #     self, llm_pipeline: vLLMPipeline, dendrite: bt.dendrite, metagraph: bt.metagraph, update_scores: callable
    # ) -> dict[str, Any]:
    #     timer_total = time.perf_counter()

    #     timer_sample = time.perf_counter()
    #     if is_organic_sample := (not self._organic_queue.is_empty()):
    #         # Choose organic sample based on the organic queue logic.
    #         sample = self._organic_queue.sample()
    #     elif self._synth_dataset is not None:
    #         # Choose if organic queue is empty, choose random sample from provided datasets.
    #         sample = random.choice(self._synth_dataset).sample()
    #     else:
    #         return {}

    #     if sample.get("organic", False):
    #         task = OrganicTask(context=sample, reference=reference)
    #     else:
    #         task = SynthOrganicTask(context=sample, reference=reference)

    #     timer_sample_elapsed = time.perf_counter() - timer_sample

    #     # Concurrently generate reference and query miners.
    #     timer_responses = time.perf_counter()
    #     reference_task = task.generate_reference(sample, llm_pipeline)
    #     responses_task = self._query_miners(
    #         sample=sample, dendrite=dendrite, tokenizer=llm_pipeline.tokenizer, metagraph=metagraph
    #     )
    #     reference, responses = await asyncio.gather(reference_task, responses_task)
    #     timer_responses_elapsed = time.perf_counter() - timer_responses

    #     # Generate rewards.
    #     timer_rewards = time.perf_counter()
    #     reward_config = OrganicRewardConfig()
    #     reward_config.apply(responses=responses, reference=reference)

    #     # rewards = await self._generate_rewards(sample, responses, reference)
    #     # rewards
    #     timer_rewards_elapsed = time.perf_counter() - timer_rewards

    #     # Set weights based on the generated rewards.
    #     timer_weights = time.perf_counter()
    #     # await self._set_weights(rewards)
    #     await self._set_weights(
    #         reward_config, is_organic=sample["organic"], uids=responses.keys(), update_scores=update_scores
    #     )
    #     timer_weights_elapsed = time.perf_counter() - timer_weights

    #     # Log the metrics.
    #     timer_elapsed = time.perf_counter() - timer_total
    #     logs = {
    #         "organic_time_sample": timer_sample_elapsed,
    #         "organic_time_responses": timer_responses_elapsed,
    #         "organic_time_rewards": timer_rewards_elapsed,
    #         "organic_time_weights": timer_weights_elapsed,
    #         "organic_time_total": timer_elapsed,
    #         "organic_queue_size": self._organic_queue.size,
    #         "is_organic_sample": is_organic_sample,
    #     }
    #     return await self._log_results(
    #         logs=logs,
    #         reference=reference,
    #         responses=responses,
    #         rewards=reward_config.final_rewards,
    #         sample=sample,
    #     )

    async def _generate_rewards(
        self, sample: SynthDatasetEntry, responses: dict[str, SynapseStreamResult], reference: str
    ):
        return {
            "rewards": OrganicRewardConfig.apply(responses=responses, reference=reference, query=sample.messages[-1]),
            "uids": responses.keys(),
            "organic": sample.organic,
        }

    @override
    async def _priority_fn(self, synapse: StreamPromptingSynapse) -> float:
        """Priority function for the axon."""
        return 10000000.0

    @override
    async def _blacklist_fn(self, synapse: StreamPromptingSynapse) -> Tuple[bool, str]:
        """Blacklist function for the axon."""
        # ! DO NOT CHANGE `Tuple` return type to `tuple`, it will break the code (bittensor internal signature checks).
        # We expect the API to be run with one specific hotkey (e.g. OTF).
        return synapse.dendrite.hotkey != settings.ORGANIC_WHITELIST_HOTKEY, ""

    @override
    async def _on_organic_entry(
        self, synapse: StreamPromptingSynapse, metagraph: bt.metagraph, wallet: bt.wallet
    ) -> StreamPromptingSynapse:
        """Organic query handle."""
        bt.logging.info(f"[Organic] Received from {synapse.dendrite.hotkey}, IP: {synapse.dendrite.ip}")

        uids = list(self.get_random_uids())
        completions: dict[int, dict] = {}
        token_streamer = partial(
            self._stream_miner_response,
            synapse,
            uids,
            completions,
            metagraph=metagraph,
            wallet=wallet,
        )

        streaming_response = synapse.create_streaming_response(token_streamer)
        self._organic_queue.add(
            {
                "roles": synapse.roles,
                "messages": synapse.messages,
                "organic": True,
                "synapse": synapse,
                "streaming_response": streaming_response,
                "uids": uids,
                "completions": completions,
            }
        )
        return streaming_response

    async def _stream_miner_response(
        self,
        synapse: StreamPromptingSynapse,
        uids: list[int],
        completions: dict[int, dict],
        send: Send,
    ):
        """Stream back miner's responses."""
        bt.logging.info(f"[Organic] Querying miner UIDs: {uids}")
        try:
            async with dendrite(wallet=self.wallet) as dend:
                responses = await dend(
                    axons=[self.metagraph.axons[uid] for uid in uids],
                    synapse=synapse,
                    timeout=settings.ORGANIC_TIMEOUT,
                    deserialize=False,
                    streaming=True,
                )
        except Exception as e:
            bt.logging.error(f"[Organic] Error querying dendrite: {e}")
            return

        async def stream_miner_chunks(uid: int, chunks: AsyncGenerator):
            accumulated_chunks: list[str] = []
            accumulated_chunks_timings: list[float] = []
            accumulated_tokens_per_chunk: list[int] = []
            synapse: StreamPromptingSynapse | None = None
            completions[uid] = {"completed": False}
            timer_start = time.perf_counter()
            async for chunk in chunks:
                try:
                    if isinstance(chunk, str):
                        accumulated_chunks.append(chunk)
                        accumulated_chunks_timings.append(time.perf_counter() - timer_start)
                        json_chunk = json.dumps({"uid": uid, "chunk": chunk})
                        await send(
                            {
                                "type": "http.response.body",
                                "body": json_chunk.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                    elif isinstance(chunk, StreamPromptingSynapse):
                        synapse = chunk
                except Exception as e:
                    bt.logging.error(f"[Organic] Error while streaming chunks: {e}")
                    break
            # TODO: Do we need to identify the end of each miner's response?
            # json_chunk = json.dumps({"uid": uid, "chunk": b"", "completed": True})
            # await send({"type": "http.response.body", "body": json_chunk, "more_body": False})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            completions[uid]["accumulated_chunks"] = accumulated_chunks
            completions[uid]["accumulated_chunks_timings"] = accumulated_chunks_timings
            completions[uid]["accumulated_tokens_per_chunk"] = accumulated_tokens_per_chunk
            completions[uid]["completed"] = True
            completions[uid]["synapse"] = synapse
            # bt.logging.debug(f"[Organic] Streaming {uid}: {''.join(accumulated_chunks)}")

        bt.logging.info(f"[Organic] Awaiting miner streams UIDs: {uids}")
        await asyncio.gather(
            *[stream_miner_chunks(uid, chunks) for uid, chunks in zip(uids, responses)],
            return_exceptions=True,
        )

    async def _reuse_organic_response(self, sample: SynthDatasetEntry) -> dict[int, SynapseStreamResult]:
        """Return a dictionary where the keys are miner UIDs and the values are their corresponding streaming responses.

        This method reuses miner responses for organic data. It waits for each miner to complete within the
        `neuron.organic_timeout` specified timeout and returns the responses. For miners who exceed the timeout,
        an empty synapse response is returned.

        Args:
            sample: Dict where the keys are miner UIDs and the values are the input streaming synapses.
        """
        if not sample.organic:
            return None

        uids = sample["uids"]
        responses: dict[int, SynapseStreamResult] = {}
        bt.logging.info(f"[Organic] Reusing miner responses for organic data, UIDs: {uids}")

        async def _check_completion(sample: dict[str, Any], uid: int):
            while not sample["completions"][uid]["completed"]:
                await asyncio.sleep(0.1)

        async def _wait_for_completion(uid: int):
            try:
                await asyncio.wait_for(
                    _check_completion(sample, uid),
                    settings.ORGANIC_TIMEOUT,
                )
                response = SynapseStreamResult(
                    accumulated_chunks=sample["completions"][uid]["accumulated_chunks"],
                    accumulated_chunks_timings=sample["completions"][uid]["accumulated_chunks_timings"],
                    tokens_per_chunk=sample["completions"][uid]["accumulated_tokens_per_chunk"],
                    synapse=sample["completions"][uid]["synapse"],
                    uid=uid,
                    exception=None,
                )
            except asyncio.TimeoutError:
                response = SynapseStreamResult(
                    accumulated_chunks=[],
                    accumulated_chunks_timings=[],
                    tokens_per_chunk=[],
                    synapse=None,
                    uid=uid,
                    exception=None,
                )
            responses[uid] = response

        await asyncio.gather(*[_wait_for_completion(uid) for uid in uids])
        return responses

    @override
    async def _query_miners(self, sample: SynthDatasetEntry) -> dict[str, SynapseStreamResult]:
        """Query miners with the given synthetic or organic sample."""
        # if sample.organic and not settings.ORGANIC_REUSE_RESPONSE_DISABLED:
        #     responses = await self._reuse_organic_response(sample)
        #     return responses

        # Get the list of uids to query.
        uids = self.get_random_uids()
        bt.logging.info(f"[Organic] Querying miners with synthetic data, UIDs: {uids}")
        streams_responses = await dendrite.forward(
            axons=[self.metagraph.axons[uid] for uid in uids],
            synapse=StreamPromptingSynapse(roles=sample.roles, messages=sample.messages),
            timeout=settings.ORGANIC_TIMEOUT,
            deserialize=False,
            streaming=True,
        )
        stream_results_dict = dict(zip(uids, streams_responses))
        responses = await handle_response(stream_results_dict, tokenizer=self.tokenizer)
        return dict(zip(uids, responses))

    @override
    async def _set_weights(self, reward_result: dict[str, Any]):
        """Set weights based on the given reward."""
        if not reward_result.get("organic", False):
            reward_result["rewards"] *= settings.ORGANIC_SYNTH_REWARD_SCALE

        # uids_to_reward = dict(zip(reward_result["uids"], reward_result["rewards"]))
        # self.update_scores(reward_result["rewards"], reward_result["uids"])

    # @override
    async def _generate_reference(self, sample: dict[str, Any]) -> str:
        """Generate reference for the given organic or synthetic sample."""
        _, reference = OrganicTask.generate_reference(sample, self.llm_pipeline)
        return reference
