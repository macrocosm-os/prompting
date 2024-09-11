from loguru import logger
from prompting.settings import settings
from prompting.base.protocol import StreamPromptingSynapse
from starlette.types import Send
from typing import Tuple, AsyncGenerator
import time
import json
import asyncio
from prompting.tasks.inference import InferenceTask
import bittensor as bt
from bittensor.dendrite import dendrite
from prompting.rewards.scoring import task_scorer
from prompting.datasets.base import ChatEntry
from prompting.utils.uids import get_random_uids
from prompting.base.forward import SynapseStreamResult
from prompting.base.dendrite import DendriteResponseEvent
from functools import partial
from dataclasses import dataclass, field


@dataclass
class Completion:
    uid: int
    completed: bool = False
    accumulated_chunks: list[str] = field(default_factory=list)
    accumulated_chunks_timings: list[float] = field(default_factory=list)
    accumulated_tokens_per_chunk: list[int] = field(default_factory=list)


async def priority_fn(synapse: StreamPromptingSynapse) -> float:
    """Priority function for the axon."""
    return 10000000.0


async def blacklist_fn(synapse: StreamPromptingSynapse) -> Tuple[bool, str]:
    """Blacklist function for the axon."""
    # ! DO NOT CHANGE `Tuple` return type to `tuple`, it will break the code (bittensor internal signature checks).
    # We expect the API to be run with one specific hotkey (e.g. OTF).
    return synapse.dendrite.hotkey != settings.ORGANIC_WHITELIST_HOTKEY, ""


async def on_organic_entry(synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
    """Organic query handle."""
    if not isinstance(synapse, StreamPromptingSynapse) or synapse.task_name != InferenceTask.__name__:
        logger.error(f"[Organic] Received non-inference task: {synapse.task_name}")
        return
    logger.info(f"[Organic] Received from {synapse.dendrite.hotkey}, IP: {synapse.dendrite.ip}")
    task = InferenceTask(query=synapse.messages[-1], messages=synapse.messages)
    miner_synapse = StreamPromptingSynapse(
        task_name=task.__class__.__name__,
        seed=task.seed,
        target_model=task.llm_model_id,
        roles=["user"],
        messages=task.messages,
    )
    # TODO: Query one of the top N incentive miners, keep the rest random.
    uids = list(get_random_uids(k=settings.ORGANIC_SAMPLE_SIZE))
    # completions: dict[int, dict] = {}
    completions: list[Completion] = []
    token_streamer = partial(
        stream_miner_response,
        miner_synapse,
        uids,
        completions,
    )

    streaming_response = miner_synapse.create_streaming_response(token_streamer)
    logger.debug(f"Message: {miner_synapse.messages}; Completions: {completions}")
    asyncio.create_task(wait_and_add(task, completions, synapse=synapse, uids=uids))
    return streaming_response


async def wait_and_add(
    task: InferenceTask, completions: list[Completion], synapse: StreamPromptingSynapse, uids: list[int]
):
    logger.debug("[ORGANIC] Waiting for responses to be collected")

    async def wait_for_responses():
        while len(completions) < len(uids):
            await asyncio.sleep(0.1)

    try:
        await asyncio.wait_for(wait_for_responses(), timeout=settings.ORGANIC_TIMEOUT)
        logger.debug(f"[ORGANIC] All {len(uids)} responses collected successfully.")
    except asyncio.TimeoutError:
        logger.error("[ORGANIC] Some responses couldn't be collected in time...")

    if len(completions) == 0:
        logger.error("[ORGANIC] No responses collected...")
        return
    logger.debug(f"[ORGANIC] Responses collected. Completions: {completions}")
    stream_results = [
        SynapseStreamResult(
            accumulated_chunks=completion.accumulated_chunks,
            accumulated_chunks_timings=(completion.accumulated_chunks_timings),
            synapse=synapse,
            uid=completion.uid,
        )
        for completion in completions
    ]
    logger.debug(f"[ORGANIC] Number of responses collected: {len(completions)}")
    response_event = DendriteResponseEvent(
        uids=uids,
        stream_results=stream_results,
        timeout=settings.NEURON_TIMEOUT,
    )
    logger.debug("[ORGANIC] Adding to scoring queue")
    task_scorer.add_to_queue(
        task=task,
        response=response_event,
        dataset_entry=ChatEntry(messages=synapse.messages, roles=synapse.roles, organic=True, source=None),
    )


async def stream_miner_response(
    synapse: StreamPromptingSynapse,
    uids: list[int],
    completions: list[Completion],
    send: Send,
):
    """Stream back miner's responses."""

    async def stream_miner_chunks(uid: int, chunks: AsyncGenerator):
        logger.debug(f"[ORGANIC] Streaming chunks for UID: {uid}")
        accumulated_chunks: list[str] = []
        accumulated_chunks_timings: list[float] = []
        accumulated_tokens_per_chunk: list[int] = []
        synapse: StreamPromptingSynapse | None = None
        timer_start = time.perf_counter()
        async for chunk in chunks:
            try:
                if isinstance(chunk, str):
                    accumulated_chunks.append(chunk)
                    accumulated_chunks_timings.append(time.perf_counter() - timer_start)
                    json_chunk = json.dumps({"uid": int(uid), "chunk": chunk})
                    await send(
                        {
                            "type": "http.response.body",
                            "body": json_chunk.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                elif isinstance(chunk, StreamPromptingSynapse):
                    synapse = chunk
            except Exception:
                logger.exception("[Organic] Error while streaming chunks")
                break
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        logger.debug(f"[ORGANIC] Appending completion for UID: {uid}")
        completions.append(
            Completion(
                uid=uid,
                accumulated_chunks=accumulated_chunks,
                accumulated_chunks_timings=accumulated_chunks_timings,
                accumulated_tokens_per_chunk=accumulated_tokens_per_chunk,
                completed=True,
                synapse=synapse,
            )
        )

    logger.info(f"[Organic] Querying miner UIDs: {uids}")
    try:
        async with dendrite(wallet=settings.WALLET) as dend:
            responses = await dend(
                axons=[settings.METAGRAPH.axons[uid] for uid in uids],
                synapse=synapse,
                timeout=settings.ORGANIC_TIMEOUT,
                deserialize=False,
                streaming=True,
            )
    except Exception as e:
        logger.exception(f"[Organic] Error querying dendrite: {e}")
        return

    logger.info(f"[Organic] Awaiting miner streams UIDs: {uids}")
    await asyncio.gather(
        *[stream_miner_chunks(uid, chunks) for uid, chunks in zip(uids, responses)],
        return_exceptions=True,
    )


def start_organic(axon: bt.axon):
    axon.attach(
        forward_fn=on_organic_entry,
        blacklist_fn=blacklist_fn,
        priority_fn=priority_fn,
    )
