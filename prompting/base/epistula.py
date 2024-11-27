import asyncio
import json
import time
from hashlib import sha256
from math import ceil
from typing import Annotated, Any, Dict, List, Optional
from uuid import uuid4

import bittensor as bt
import httpx
import openai
from httpx import Timeout
from loguru import logger
from substrateinterface import Keypair

from prompting.base.dendrite import SynapseStreamResult
from prompting.settings import settings


def verify_signature(
    signature, body: bytes, timestamp, uuid, signed_for, signed_by, now
) -> Optional[Annotated[str, "Error Message"]]:
    if not isinstance(signature, str):
        return "Invalid Signature"
    timestamp = int(timestamp)
    if not isinstance(timestamp, int):
        return "Invalid Timestamp"
    if not isinstance(signed_by, str):
        return "Invalid Sender key"
    if not isinstance(signed_for, str):
        return "Invalid receiver key"
    if not isinstance(uuid, str):
        return "Invalid uuid"
    if not isinstance(body, bytes):
        return "Body is not of type bytes"
    ALLOWED_DELTA_MS = 8000
    keypair = Keypair(ss58_address=signed_by)
    if timestamp + ALLOWED_DELTA_MS < now:
        return "Request is too stale"
    message = f"{sha256(body).hexdigest()}.{uuid}.{timestamp}.{signed_for}"
    verified = keypair.verify(message, signature)
    if not verified:
        return "Signature Mismatch"
    return None


def generate_header(
    hotkey: Keypair,
    body_bytes: Dict[str, Any],
    signed_for: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = round(time.time() * 1000)
    timestampInterval = ceil(timestamp / 1e4) * 1e4
    uuid = str(uuid4())
    headers = {
        "Epistula-Version": "2",
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Uuid": uuid,
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Request-Signature": "0x"
        + hotkey.sign(f"{sha256(body_bytes).hexdigest()}.{uuid}.{timestamp}.{signed_for or ''}").hex(),
    }
    if signed_for:
        headers["Epistula-Signed-For"] = signed_for
        headers["Epistula-Secret-Signature-0"] = "0x" + hotkey.sign(str(timestampInterval - 1) + "." + signed_for).hex()
        headers["Epistula-Secret-Signature-1"] = "0x" + hotkey.sign(str(timestampInterval) + "." + signed_for).hex()
        headers["Epistula-Secret-Signature-2"] = "0x" + hotkey.sign(str(timestampInterval + 1) + "." + signed_for).hex()
        headers.update(json.loads(body_bytes))
    return headers


def create_header_hook(hotkey, axon_hotkey):
    async def add_headers(request: httpx.Request):
        for key, header in generate_header(hotkey, request.read(), axon_hotkey).items():
            if key not in ["messages", "model", "stream"]:
                request.headers[key] = str(header)
        return request

    return add_headers


async def query_miners(uids: list = [], body: bytes = b"", stream: bool = False):
    try:
        tasks = []
        for uid in uids:
            tasks.append(
                asyncio.create_task(
                    handle_inference(
                        settings.METAGRAPH,
                        settings.WALLET,
                        body,
                        uid,
                        stream=stream,
                    )
                )
            )
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions from responses
        exceptions = [resp for resp in responses if isinstance(resp, Exception)]
        if exceptions:
            for exc in exceptions:
                logger.error(f"Error in handle_inference: {exc}")
            # Handle exceptions as needed

        if stream:
            # 'responses' is a list of async iterators (chat objects)
            async def merged_stream():
                streams = [response.__aiter__() for response in responses if not isinstance(response, Exception)]
                pending = {}
                for stream in streams:
                    try:
                        task = asyncio.create_task(stream.__anext__())
                        pending[task] = stream
                    except StopAsyncIteration:
                        continue  # Skip empty streams

                while pending:
                    done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        stream = pending.pop(task)
                        try:
                            result = task.result()
                            yield result
                            # Schedule the next item from the same stream
                            next_task = asyncio.create_task(stream.__anext__())
                            pending[next_task] = stream
                        except StopAsyncIteration:
                            # Stream is exhausted
                            pass
                        except Exception as e:
                            logger.error(f"Error while streaming: {e}")

            return merged_stream()
        else:
            # 'responses' is a list of SynapseStreamResult objects
            return [resp for resp in responses if not isinstance(resp, Exception)]
    except Exception as e:
        logger.error(f"Error in query_miners: {e}")
        return []


async def query_availabilities(uids, task_config, model_config):
    """Query the availability of the miners"""
    availability_dict = {"task_availabilities": task_config, "llm_model_availabilities": model_config}
    # Query the availability of the miners
    try:
        tasks = []
        for uid in uids:
            tasks.append(
                asyncio.create_task(
                    handle_availability(
                        settings.METAGRAPH,
                        availability_dict,
                        uid,
                    )
                )
            )
        responses: List[SynapseStreamResult] = await asyncio.gather(*tasks)
        return responses

    except Exception as e:
        logger.error(f"Error in availability call: {e}")
        return []


async def handle_availability(
    metagraph: "bt.NonTorchMetagraph",
    request: Dict[str, Any],
    uid: int,
) -> Dict[str, bool]:
    try:
        axon_info = metagraph.axons[uid]
        url = f"http://{axon_info.ip}:{axon_info.port}/availability"

        timeout = httpx.Timeout(settings.NEURON_TIMEOUT, connect=5, read=5)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=request)

        response.raise_for_status()
        return response.json()

    except Exception:
        return {}


async def handle_inference(
    metagraph: "bt.NonTorchMetagraph",
    wallet: "bt.wallet",
    body: Dict[str, Any],
    uid: int,
    stream: bool = False,
) -> SynapseStreamResult:
    exception = None
    chunks = []
    chunk_timings = []
    try:
        start_time = time.time()
        axon_info = metagraph.axons[uid]
        miner = openai.AsyncOpenAI(
            base_url=f"http://{axon_info.ip}:{axon_info.port}/v1",
            api_key="Apex",
            max_retries=0,
            timeout=Timeout(settings.NEURON_TIMEOUT, connect=5, read=10),
            http_client=openai.DefaultAsyncHttpxClient(
                event_hooks={"request": [create_header_hook(wallet.hotkey, axon_info.hotkey)]}
            ),
        )
        payload = json.loads(body)
        chat = await miner.chat.completions.create(
            messages=payload["messages"],
            model=payload["model"],
            stream=True,
            extra_body={k: v for k, v in payload.items() if k not in ["messages", "model"]},
        )
        if not stream:
            async for chunk in chat:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
                    chunk_timings.append(time.time() - start_time)
    except openai.APIConnectionError as e:
        logger.trace(f"Miner {uid} failed request: {e}")
        exception = str(e)
    except Exception as e:
        logger.trace(f"Unknown Error when sending to miner {uid}: {e}")
        exception = str(e)
    finally:
        if exception is None:
            status_code = 200
            status_message = "Success"
        elif isinstance(exception, openai.APIConnectionError):
            status_code = 502
            status_message = exception
        else:
            status_code = 500
            status_message = exception

    if stream:
        return chat
    else:
        return SynapseStreamResult(
            accumulated_chunks=chunks,
            accumulated_chunks_timings=chunk_timings,
            uid=uid,
            exception=exception,
            status_code=status_code,
            status_message=status_message,
        )
