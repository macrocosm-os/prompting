import os
import csv
import asyncio
import json
import time
from functools import partial
from typing import Any, AsyncGenerator, Literal, Sequence, Tuple, Union

import bittensor as bt
import torch
import numpy as np
from organic_scoring import OrganicScoringBase
from organic_scoring.synth_dataset import SynthDatasetBase
from starlette.types import Send
from typing_extensions import override

from prompting.agent import HumanAgent
from prompting.base.neuron import BaseNeuron
from prompting.cleaners.cleaner import CleanerPipeline
from prompting.dendrite import DendriteResponseEvent, SynapseStreamResult
from prompting.forward import handle_response
from prompting.llms.vllm_llm import vLLM_LLM
from prompting.organic.organic_task import OrganicTask
from prompting.organic.synth_organic_task import SynthOrganicTask
from prompting.protocol import StreamPromptingSynapse
from prompting.rewards.pipeline import RewardPipeline
from prompting.rewards.reward import RewardResult
from prompting.tasks.task import make_system_prompt
from prompting.utils.logging import log_event
from prompting.utils.uids import get_random_uids, get_uids


class OrganicScoringPrompting(OrganicScoringBase):
    def __init__(
        self,
        axon: bt.axon,
        synth_dataset: Union[SynthDatasetBase, Sequence[SynthDatasetBase]],
        trigger_frequency: Union[float, int],
        trigger: Literal["seconds", "steps"],
        validator: BaseNeuron,
        trigger_frequency_min: Union[float, int] = 5,
        trigger_scaling_factor: Union[float, int] = 50,
    ):
        """Organic Scoring implementation.

        Organic scoring runs in a separate `asyncio` task and is triggered by a timer or a step counter.

        Process Workflow:
        1. Trigger Check: Upon triggering the rewarding process, the system checks if the organic queue is empty.
            If the queue is empty, synthetic datasets are used to bootstrap the organic scoring mechanism.
            Otherwise, samples from the organic queue are utilized.
        2. Data Processing: The sampled data is concurrently passed to the `_query_miners` and `_generate_reference`
            methods.
        3. Reward Generation: After receiving responses from miners and any reference data, the information
            is processed by the `_generate_rewards` method.
        4. Weight Setting: The generated rewards are then applied through the `_set_weights` method.
        5. Logging: Finally, the results can be logged using the `_log_results` method, along with all relevant data
            provided as arguments, and default time elapsed on each step of rewarding process.
        """
        super().__init__(
            axon=axon,
            synth_dataset=synth_dataset,
            trigger_frequency=trigger_frequency,
            trigger=trigger,
            trigger_frequency_min=trigger_frequency_min,
            trigger_scaling_factor=trigger_scaling_factor,
        )
        self._val = validator
        # Organic scoring reward pipeline.
        self._reward_pipeline = RewardPipeline(
            selected_tasks=[OrganicTask.name, SynthOrganicTask.name],
            device=self._val.device,
            available_tasks={
                OrganicTask.name: OrganicTask,
                SynthOrganicTask.name: SynthOrganicTask,
            },
        )
        # Debugging CSV.
        self._synth_file = "synth.csv"
        self._organic_file = "organic.csv"
        self._fieldnames = [
            "turn",
            "total_rewards",
            "chosen_uid",
            "message",
            "reference",
            "chosen_response",
        ]
        file_exists = os.path.isfile(self._organic_file)

        with open(self._organic_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, self._fieldnames)
            if not file_exists:
                writer.writeheader()

        file_exists = os.path.isfile(self._synth_file)
        with open(self._synth_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, self._fieldnames)
            if not file_exists:
                writer.writeheader()

    @override
    async def _priority_fn(self, synapse: StreamPromptingSynapse) -> float:
        """Priority function for the axon."""
        return 10000000.0

    @override
    async def _blacklist_fn(self, synapse: StreamPromptingSynapse) -> Tuple[bool, str]:
        """Blacklist function for the axon."""
        # ! DO NOT CHANGE `Tuple` return type to `tuple`, it will break the code (bittensor internal signature checks).
        # We expect the API to be run with one specific hotkey (e.g. OTF).
        return synapse.dendrite.hotkey != self._val.config.neuron.organic_whitelist_hotkey, ""

    @override
    async def _on_organic_entry(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        """Organic query handle."""
        bt.logging.info(f"[Organic] Received from {synapse.dendrite.hotkey}, IP: {synapse.dendrite.ip}")

        uids = get_uids(
            self._val,
            sampling_mode=self._val.config.neuron.organic_sampling_mode,
            k=self._val.config.neuron.organic_sample_size,
            exclude=[],
        )
        uids_list = uids.cpu().tolist()
        completions: dict[int, dict] = {}
        token_streamer = partial(
            self._stream_miner_response,
            synapse,
            uids_list,
            completions,
        )

        streaming_response = synapse.create_streaming_response(token_streamer)
        self._organic_queue.add(
            {
                "roles": synapse.roles,
                "messages": synapse.messages,
                "organic": True,
                "synapse": synapse,
                "streaming_response": streaming_response,
                "uids": uids_list,
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
        responses = self._val.dendrite.query(
            axons=[self._val.metagraph.axons[uid] for uid in uids],
            synapse=synapse,
            timeout=self._val.config.neuron.organic_timeout,
            deserialize=False,
            streaming=True,
        )

        async def stream_miner_chunks(uid: int, chunks: AsyncGenerator):
            accumulated_chunks: list[str] = []
            accumulated_chunks_timings: list[float] = []
            accumulated_tokens_per_chunk: list[int] = []
            synapse: StreamPromptingSynapse | None = None
            completions[uid] = {"completed": False}
            timer_start = time.perf_counter()
            async for chunk in chunks:
                if isinstance(chunk, str):
                    accumulated_chunks.append(chunk)
                    accumulated_chunks_timings.append(time.perf_counter() - timer_start)
                    accumulated_tokens_per_chunk.append(len(self._val.llm_pipeline.tokenizer.tokenize(chunk)))
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
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            completions[uid]["accumulated_chunks"] = accumulated_chunks
            completions[uid]["accumulated_chunks_timings"] = accumulated_chunks_timings
            completions[uid]["accumulated_tokens_per_chunk"] = accumulated_tokens_per_chunk
            completions[uid]["completed"] = True
            completions[uid]["synapse"] = synapse
            bt.logging.debug(f"[Organic] Streaming back {uid}: {''.join(accumulated_chunks)}")

        bt.logging.info(f"[Organic] Awaiting miner streams UIDs: {uids}")
        await asyncio.gather(*[stream_miner_chunks(uid, chunks) for uid, chunks in zip(uids, responses)])

    async def _reuse_organic_response(self, sample: dict[str, Any]) -> dict[int, SynapseStreamResult]:
        """Return a dictionary where the keys are miner UIDs and the values are their corresponding streaming responses.

        This method reuses miner responses for organic data. It waits for each miner to complete within the
        `neuron.organic_timeout` specified timeout and returns the responses. For miners who exceed the timeout,
        an empty synapse response is returned.

        Args:
            sample: Dict where the keys are miner UIDs and the values are the input streaming synapses.
        """
        if not sample.get("organic", False):
            return None

        uids_cpu = sample["uids"]
        responses: dict[int, SynapseStreamResult] = {}
        bt.logging.info(f"[Organic] Reusing miner responses for organic data, UIDs: {uids_cpu}")

        async def _check_completion(sample: dict[str, Any], uid: int):
            while not sample["completions"][uid]["completed"]:
                await asyncio.sleep(0.1)

        async def _wait_for_completion(uid: int):
            try:
                await asyncio.wait_for(
                    _check_completion(sample, uid),
                    self._val.config.neuron.organic_timeout,
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

        await asyncio.gather(*[_wait_for_completion(uid) for uid in uids_cpu])
        return responses

    @override
    async def _query_miners(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Query miners with the given synthetic or organic sample."""
        if sample.get("organic", False) and not self._val.config.neuron.organic_reuse_response_disabled:
            responses = await self._reuse_organic_response(sample)
            return responses

        # Get the list of uids to query.
        uids = get_random_uids(self._val, k=self._val.config.neuron.organic_sample_size, exclude=None).to(
            self._val.device
        )
        uids_cpu = uids.cpu().tolist()
        bt.logging.info(f"[Organic] Querying miners with synthetic data, UIDs: {uids_cpu}")
        streams_responses = self._val.dendrite.query(
            axons=[self._val.metagraph.axons[uid] for uid in uids_cpu],
            synapse=StreamPromptingSynapse(roles=sample["roles"], messages=sample["messages"]),
            timeout=self._val.config.neuron.organic_timeout,
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
        """Generate rewards for the given sample, responses, and reference."""
        assert reference is not None
        if sample.get("organic", False):
            task = OrganicTask(context=sample, reference=reference)
        else:
            task = SynthOrganicTask(context=sample, reference=reference)
        stream_results = list(responses.values())
        uids_list = list(responses.keys())
        uids = torch.tensor(uids_list)
        timeout = self._val.config.neuron.organic_timeout
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
            self._reward_pipeline,
            agent=agent,
            response_event=response_event,
            device=self._val.device,
        )
        bt.logging.info(f"[Organic] RewardResult: {reward_result}")
        return {"reward": reward_result, "uids": uids_list, "agent": agent}

    @override
    async def _set_weights(self, reward: dict[str, Any]):
        """Set weights based on the given reward."""
        uids = reward["uids"]
        reward_result = reward["reward"]
        bt.logging.info(f"[Organic] Rewards for miner's UIDs: {dict(zip(uids, reward_result.rewards))}")
        bt.logging.info(f"[Organic] Weight setting enabled: {self._val.config.neuron.organic_set_weights_enabled}")
        if self._val.config.neuron.organic_set_weights_enabled:
            self._val.update_scores(reward_result.rewards, uids)
            # Sync is not needed as it's done in the benchmarks loop.
            # self._val.sync()

    @override
    async def _log_results(
        self,
        logs: dict[str, Any],
        reference: str,
        responses: dict[int, SynapseStreamResult],
        rewards: dict[str, Any],
        sample: dict[str, Any],
        *args,
        **kwargs,
    ):
        logs["block"] = self._val.block
        logs["step"] = self._val.step
        # Length of messages is incremented by 2 every step: query and response.
        logs["turn"] = len(sample["messages"]) // 2
        completions_len: list[int] = [len(response.synapse.completion) for response in responses.values()]
        logs["organic_response_mean_chars"] = np.mean(completions_len)
        logs["organic_response_std_chars"] = np.std(completions_len)
        logs["organic_reference_chars"] = len(reference)
        logs.update(rewards["reward"].__state_dict__(full=self._val.config.neuron.log_full))
        log_event(self._val, logs)

        def write(file: str):
            with open(file, mode="a", newline="") as file:
                writer = csv.DictWriter(file, self._fieldnames)
                reward_values: list[float] = rewards["reward"].rewards.tolist()
                writer.writerow(
                    {
                        "turn": logs["turn"],
                        "total_rewards": [reward for reward in reward_values],
                        "chosen_uid": next(iter(responses.keys())),
                        "message": sample["messages"][-1].replace("\n", "--"),
                        "reference": reference.replace("\n", "--"),
                        "chosen_response": next(iter(responses.values())).synapse.completion.replace("\n", "--"),
                    }
                )

        if sample.get("organic", False):
            write(self._organic_file)
        else:
            write(self._synth_file)

        return logs

    @override
    async def _generate_reference(self, sample: dict[str, Any]) -> str:
        """Generate reference for the given organic or synthetic sample."""
        async with self._val.lock:
            reference = vLLM_LLM(
                self._val.llm_pipeline,
                system_prompt=make_system_prompt(),
                max_new_tokens=self._val.config.neuron.organic_reference_max_tokens,
            ).query_conversation(
                messages=sample["messages"],
                roles=sample["roles"],
                cleaner=CleanerPipeline(cleaning_pipeline=[]),
            )
        return reference
