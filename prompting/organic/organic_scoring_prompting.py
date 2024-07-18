import time
from functools import partial
from typing import Any, Literal, Sequence

import bittensor as bt
import torch
from organic_scoring import OrganicScoringBase
from organic_scoring.synth_dataset import SynthDatasetBase
from prompting.agent import HumanAgent
from prompting.base.neuron import BaseNeuron
from prompting.cleaners.cleaner import CleanerPipeline
from prompting.dendrite import DendriteResponseEvent, SynapseStreamResult
from prompting.forward import handle_response
from prompting.llms.vllm_llm import vLLM_LLM
from prompting.organic import organic_task
from prompting.protocol import StreamPromptingSynapse
from prompting.rewards.reward import RewardResult
from prompting.tasks.task import make_system_prompt
from prompting.utils.logging import log_event
from prompting.utils.uids import get_random_uids, get_uids
from starlette.types import Send
from typing_extensions import override


class OrganicScoringPrompting(OrganicScoringBase):
    def __init__(
        self,
        axon: bt.axon,
        synth_dataset: SynthDatasetBase | Sequence[SynthDatasetBase],
        trigger_frequency: float | int,
        trigger: Literal["seconds", "steps"],
        validator: BaseNeuron,
    ):
        super().__init__(axon=axon, synth_dataset=synth_dataset, trigger_frequency=trigger_frequency, trigger=trigger)
        self._val = validator

    async def _priority_fn(self, synapse: StreamPromptingSynapse) -> float:
        """Priority function for the axon"""
        return 1000000.0

    async def _blacklist_fn(self, synapse: StreamPromptingSynapse) -> tuple[bool, str]:
        """Blacklist function for the axon"""
        return False, ""

    @override
    async def _on_organic_entry(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        bt.logging.info(f"[Organic] Received from {synapse.dendrite.hotkey}, IP: {synapse.dendrite.ip}")

        uids = get_uids(
            self._val,
            sampling_mode=self._val.config.neuron.organic_sampling_mode,
            k=self._val.config.neuron.organic_size,
            exclude=[])
        uids_list = uids.cpu().tolist()
        completions: dict[int, dict] = {}
        token_streamer = partial(
            self._stream_miner_response,
            synapse,
            uids_list,
            completions,
        )

        streaming_response = synapse.create_streaming_response(token_streamer)
        self._organic_queue.add({
            "roles": synapse.roles,
            "messages": synapse.messages,
            "organic": True,
            "synapse": synapse,
            "streaming_response": streaming_response,
            "uids": uids_list,
            "completions": completions,
        })
        return streaming_response

    async def _stream_miner_response(
        self,
        synapse: StreamPromptingSynapse,
        uids: list[int],
        completions: dict[int, dict],
        send: Send,
    ):
        bt.logging.info(f"[Organic] Querying miner UIDs: {uids}")
        responses = self._val.dendrite.query(
            axons=[self._val.metagraph.axons[uid] for uid in uids],
            synapse=synapse,
            timeout=self._val.config.neuron.organic_timeout,
            deserialize=False,
            streaming=True,
        )

        bt.logging.info(f"[Organic] Awaiting miner streams UIDs: {uids}")
        for uid, chunks in zip(uids, responses):
            accumulated_chunks: list[str] = []
            accumulated_chunks_timings: list[float] = []
            accumulated_tokens_per_chunk: list[int] = []
            completions[uid] = {"completed": False}
            timer_start = time.perf_counter()
            async for chunk in chunks:
                if isinstance(chunk, str):
                    accumulated_chunks.append(chunk)
                    accumulated_chunks_timings.append(time.perf_counter() - timer_start)
                    accumulated_tokens_per_chunk.append(len(self._val.llm_pipeline.tokenizer.tokenize(chunk)))
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                elif isinstance(chunk, StreamPromptingSynapse):
                    synapse = chunk
            completions[uid]["accumulated_chunks"] = accumulated_chunks
            completions[uid]["accumulated_chunks_timings"] = accumulated_chunks_timings
            completions[uid]["accumulated_tokens_per_chunk"] = accumulated_tokens_per_chunk
            completions[uid]["completed"] = True
            completions[uid]["synapse"] = synapse
    
    async def _reuse_organic_response(self, sample: dict[str, Any]) -> dict[int, SynapseStreamResult]:
        """Returns a dict where the keys are miner UIDs and the values are their corresponding streaming responses"""
        if not sample.get("organic", False):
            return None
        uids_cpu = sample["uids"]
        responses: dict[int, SynapseStreamResult] = {}
        bt.logging.info(f"[Organic] Reusing miner responses for organic data, UIDs: {uids_cpu}")
        for uid in uids_cpu:
            response = SynapseStreamResult(
                accumulated_chunks=sample["completions"][uid]["accumulated_chunks"],
                accumulated_chunks_timings=sample["completions"][uid]["accumulated_chunks_timings"],
                tokens_per_chunk=sample["completions"][uid]["accumulated_tokens_per_chunk"],
                synapse=sample["completions"][uid]["synapse"],
                uid=uid,
                exception=None
            )
            responses[uid] = response
        return responses

    @override
    async def _query_miners(self, sample: dict[str, Any]) -> dict[str, Any]:
        if sample.get("organic", False):
            responses = await self._reuse_organic_response(sample)
            return responses

        # Get the list of uids to query.
        uids = get_random_uids(self._val, k=self._val.config.neuron.organic_size, exclude=None).to(self._val.device)
        uids_cpu = uids.cpu().tolist()
        bt.logging.info(f"[Organic] Querying miners with synthetic data, UIDs: {uids_cpu}")
        streams_responses = self._val.dendrite.query(
            axons=[self._val.metagraph.axons[uid] for uid in uids_cpu],
            synapse=StreamPromptingSynapse(roles=sample["roles"], messages=sample["messages"]),
            timeout=self._val.config.neuron.timeout,
            deserialize=False,
            streaming=True,
        )
        stream_results_dict = dict(zip(uids_cpu, streams_responses))
        responses = await handle_response(stream_results_dict, self._val.llm_pipeline.tokenizer)
        return dict(zip(uids_cpu, responses))

    @override
    async def _generate_rewards(
        self,
        sample: dict[str, Any],
        responses: dict[str, Any],
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        assert reference is not None
        if sample.get("organic", False):
            task = organic_task.OrganicTask(context=sample, reference=reference)
        else:
            task = organic_task.SynthOrganicTask(context=sample, reference=reference)
        stream_results = list(responses.values())
        uids_list = list(responses.keys())
        uids = torch.tensor(uids_list)
        timeout = self._val.config.neuron.timeout
        response_event = DendriteResponseEvent(stream_results=stream_results, uids=uids, timeout=timeout)

        bt.logging.debug(f"[Organic] Miner stream results: {stream_results}")

        # Dummy HumanAgent used to reuse existing reward pipeline.
        agent = HumanAgent(
            task=task,
            llm_pipeline=self._val.llm_pipeline,
            begin_conversation=True,
            system_prompt=make_system_prompt(),
        )
        reward_result = RewardResult(
            self._val.reward_pipeline,
            agent=agent,
            response_event=response_event,
            device=self._val.device,
        )
        bt.logging.info(f"[Organic] RewardResult: {reward_result}")
        return {
            "reward": reward_result,
            "uids": uids_list,
            "agent": agent
        }

    @override
    async def _set_weights(self, reward: dict[str, Any]):
        uids = reward["uids"]
        reward_result = reward["reward"]
        self._val.update_scores(reward_result.rewards, uids)
        self._val.sync()

    @override
    async def _log_results(
        self,
        logs: dict[str, Any],
        reference: dict[str, Any],
        responses: dict[str, Any],
        rewards: dict[str, Any],
        sample: dict[str, Any],
        *args,
        **kwargs,
    ):
        logs["block"] = self._val.block,
        logs["step"] = self._val.step,
        logs.update(rewards["reward"].__state_dict__(full=self._val.config.neuron.log_full))
        log_event(self._val, logs)
        return logs

    @override
    async def _generate_reference(self, sample: dict[str, Any]) -> dict[str, Any]:
        reference = vLLM_LLM(self._val.llm_pipeline, system_prompt=make_system_prompt()).query_conversation(
            messages=sample["messages"],
            roles=sample["roles"],
            cleaner=CleanerPipeline(cleaning_pipeline=[])
        )
        return reference
