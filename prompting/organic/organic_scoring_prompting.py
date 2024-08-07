import asyncio
import json
import time
from functools import partial
from typing import Any, AsyncGenerator, Tuple

import bittensor as bt
from prompting import settings
from organic_scoring import OrganicScoringBase
from starlette.types import Send
from typing_extensions import override
from bittensor.dendrite import dendrite

from prompting.base.dendrite import SynapseStreamResult
from neurons.forward import handle_response
from prompting.organic.organic_task import OrganicTask, OrganicRewardConfig
from prompting.base.protocol import StreamPromptingSynapse

# from prompting.rewards.reward import RewardResult


# from prompting.tasks.task import make_system_prompt
class SynthDatasetEntry:
    roles: list[str]
    messages: list[str]
    organic: bool
    source: str


class OrganicScoringPrompting(OrganicScoringBase):
    # axon: bt.axon
    # synth_dataset: Union[SynthDatasetBase, Sequence[SynthDatasetBase]]
    # llm_pipeline: vLLMPipeline
    # dendrite: bt.dendrite
    # metagraph: bt.metagraph
    # update_scores: callable
    # dendrite: bt.dendrite
    # tokenizer: PreTrainedTokenizerFast
    # metagraph: bt.metagraph
    # get_random_uids: callable
    # wallet: bt.wallet
    # _lock: asyncio.Lock
    # trigger_frequency: Union[float, int]

    # model_config = ConfigDict(arbitrary_types_allowed=True)

    # trigger_frequency_min: Union[float, int] = 5
    # trigger_scaling_factor: Union[float, int] = 5
    # trigger: Literal["seconds", "steps"]

    def __init__(self, **data):
        # super().__init__(**data)  # Pydantic init
        self.axon = data["axon"]
        self.synth_dataset = data["synth_dataset"]
        self.llm_pipeline = data["llm_pipeline"]
        self.dendrite = data["dendrite"]
        self.metagraph = data["metagraph"]
        self.update_scores = data["update_scores"]
        self.tokenizer = data["tokenizer"]
        self.get_random_uids = data["get_random_uids"]
        self.wallet = data["wallet"]
        self._lock = data["_lock"]
        self.trigger_frequency = data["trigger_frequency"]
        self.trigger_frequency_min = data["trigger_frequency_min"]
        self.trigger_scaling_factor = data["trigger_scaling_factor"]
        self.trigger = data["trigger"]

    async def _generate_rewards(
        self, sample: SynthDatasetEntry, responses: dict[str, SynapseStreamResult], reference: str
    ):
        _, _, rewards = OrganicRewardConfig.apply(responses=responses, reference=reference, query=sample.messages[-1])
        return {
            "rewards": rewards,
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
        self.update_scores(reward_result["rewards"], reward_result["uids"])

    # @override
    async def _generate_reference(self, sample: dict[str, Any]) -> str:
        """Generate reference for the given organic or synthetic sample."""
        async with self._lock:
            _, reference = OrganicTask.generate_reference(sample, self.llm_pipeline)
        return reference
