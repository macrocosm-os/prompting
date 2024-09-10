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
from prompting.base.dendrite import DendriteResponseEvent
from prompting.base.forward import handle_response


async def priority_fn(synapse: StreamPromptingSynapse) -> float:
    """Priority function for the axon."""
    return 10000000.0


async def blacklist_fn(synapse: StreamPromptingSynapse) -> Tuple[bool, str]:
    """Blacklist function for the axon."""
    # ! DO NOT CHANGE `Tuple` return type to `tuple`, it will break the code (bittensor internal signature checks).
    # We expect the API to be run with one specific hotkey (e.g. OTF).
    return synapse.dendrite.hotkey != settings.ORGANIC_WHITELIST_HOTKEY, ""


def on_organic_entry(synapse: StreamPromptingSynapse):
    logger.info(f"[Organic] Received from {synapse.dendrite.hotkey}, IP: {synapse.dendrite.ip}")
    task = InferenceTask(query=synapse.messages[-1], messages=synapse.messages)

    # Create a new synapse masquerading as an inference task.
    miner_synapse = StreamPromptingSynapse(
        task_name=task.__class__.__name__,
        seed=task.seed,
        target_model=task.llm_model_id,
        roles=["user"],
        messages=[task.query],
    )

    # TODO: Query one of the top N incentive miners, keep the rest random.
    uids = list(get_random_uids(k=settings.ORGANIC_SAMPLE_SIZE))
    axons = [settings.METAGRAPH.axons[uid] for uid in uids]

    streams_responses = asyncio.run(
        settings.DENDRITE(
            axons=axons,
            synapse=miner_synapse,
            timeout=settings.NEURON_TIMEOUT,
            deserialize=False,
            streaming=True,
        )
    )

    # Prepare the task for handling stream responses
    stream_results = asyncio.run(handle_response(stream_results_dict=dict(zip(uids, streams_responses))))

    # return streaming_response
    response_event = DendriteResponseEvent(
        uids=uids,
        stream_results=stream_results,
        timeout=settings.NEURON_TIMEOUT,
    )

    task_scorer.add_to_queue(
        task=task,
        response=response_event,
        dataset_entry=ChatEntry(messages=synapse.messages, roles=synapse.roles, organic=True, source=None),
    )

    return response_event


async def stream_miner_response(
    synapse: StreamPromptingSynapse,
    uids: list[int],
    completions: dict[int, dict],
    send: Send,
):
    """Stream back miner's responses."""
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
        completions[uid]["accumulated_chunks"] = accumulated_chunks
        completions[uid]["accumulated_chunks_timings"] = accumulated_chunks_timings
        completions[uid]["accumulated_tokens_per_chunk"] = accumulated_tokens_per_chunk
        completions[uid]["completed"] = True
        completions[uid]["synapse"] = synapse

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
