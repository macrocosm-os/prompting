import json
from hashlib import sha256
from uuid import uuid4
from math import ceil
import time
from substrateinterface import Keypair
import asyncio
import bittensor as bt
import traceback
from typing import Dict, List, Optional, Any, Annotated
from prompting.base.dendrite import SynapseStreamResult
from httpx import Timeout
import httpx
import openai
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
    return {**headers, **json.loads(body_bytes)}


def create_header_hook(hotkey, axon_hotkey=None, api_key=None):
    """
    Creates a header hook function that adds authentication headers including API key.

    Args:
        hotkey: The wallet hotkey
        axon_hotkey: Optional axon hotkey
        api_key: Optional API key for endpoint authentication

    Returns:
        Async function that adds headers to the request
    """

    async def add_headers(request: httpx.Request):
        # Add standard headers
        for key, header in generate_header(hotkey, request.read(), axon_hotkey).items():
            if key not in ["messages", "model", "stream"]:
                request.headers[key] = header

        # Add API key if provided
        if api_key:
            request.headers["api-key"] = api_key

        return request

    return add_headers


async def query_miners(task, uids, body):
    try:
        tasks = []
        for uid in uids:
            tasks.append(
                asyncio.create_task(
                    handle_inference(
                        settings.METAGRAPH,
                        settings.WALLET,
                        task,
                        body,
                        uid,
                    )
                )
            )
        responses: List[SynapseStreamResult] = await asyncio.gather(*tasks)
        return responses
    except Exception as e:
        bt.logging.error(f"Error in forward for: {e}")
        bt.logging.error(traceback.format_exc())
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
        bt.logging.error(f"Error in availability call: {e}")
        bt.logging.error(traceback.format_exc())
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
    task: str,
    body: Dict[str, Any],
    uid: int,
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
            timeout=Timeout(settings.NEURON_TIMEOUT, connect=5, read=5),
            http_client=openai.DefaultAsyncHttpxClient(
                event_hooks={"request": [create_header_hook(wallet.hotkey, axon_info.hotkey)]}
            ),
        )
        try:
            payload = json.loads(body)
            chat = await miner.chat.completions.create(
                messages=payload["messages"], model=payload["model"], stream=True
            )
            async for chunk in chat:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
                    chunk_timings.append(time.time() - start_time)

        except openai.APIConnectionError as e:
            bt.logging.trace(f"Miner {uid} failed request: {e}")
            exception = e

        except Exception as e:
            bt.logging.trace(f"Unknown Error when sending to miner {uid}: {e}")
            exception = e

    except Exception as e:
        exception = e
        bt.logging.error(f"{uid}: Error in forward for: {e}")
        bt.logging.error(traceback.format_exc())
    finally:
        if exception:
            exception = str(exception)
        if exception is None:
            status_code = 200
            status_message = "Success"
        elif isinstance(exception, openai.APIConnectionError):
            status_code = 502
            status_message = str(exception)
        else:
            status_code = 500
            status_message = str(exception)

        return SynapseStreamResult(
            accumulated_chunks=chunks,
            accumulated_chunks_timings=chunk_timings,
            uid=uid,
            exception=exception,
            status_code=status_code,
            status_message=status_message,
        )
