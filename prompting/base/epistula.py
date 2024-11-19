import json
from hashlib import sha256
from uuid import uuid4
from math import ceil
import time
from prompting.utils.timer import Timer
from substrateinterface import Keypair
import asyncio
import bittensor as bt
import math
from os import urandom
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any, Annotated
from prompting.base.dendrite import SynapseStreamResult
from httpx import Timeout
import httpx
import openai
import requests
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
        + hotkey.sign(
            f"{sha256(body_bytes).hexdigest()}.{uuid}.{timestamp}.{signed_for or ''}"
        ).hex(),
    }
    if signed_for:
        headers["Epistula-Signed-For"] = signed_for
        headers["Epistula-Secret-Signature-0"] = (
            "0x" + hotkey.sign(str(timestampInterval - 1) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-1"] = (
            "0x" + hotkey.sign(str(timestampInterval) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-2"] = (
            "0x" + hotkey.sign(str(timestampInterval + 1) + "." + signed_for).hex()
        )
    return headers

def create_header_hook(hotkey, axon_hotkey, task):
    async def add_headers(request: httpx.Request):
        for key, header in generate_header(hotkey, request.read(), axon_hotkey).items():
            request.headers[key] = header
        request.headers["Task"] = task

    return add_headers

async def query_miners(task, uids, body):
    try:
        tasks = []
        for uid in uids:
            tasks.append(
                asyncio.create_task(
                    handle_inference(
                        settings.METAGRAPH, settings.WALLET, task, body, uid,
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
    """ Query the availability of the miners """
    availability_dict = {'task_availabilities': task_config, 'llm_model_availabilities': model_config}
    # Query the availability of the miners
    try:
        tasks = []
        for uid in uids:
            tasks.append(
                asyncio.create_task(
                    handle_availability(
                        settings.METAGRAPH, availability_dict, uid,
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

    except Exception as e:
        # If the miner is not available, we will return a failure response
        bt.logging.error(f"Miner {uid} failed request: {e}")
        return {}


async def handle_inference(
    metagraph: "bt.NonTorchMetagraph",
    wallet: "bt.wallet",
    task: str,
    body: Dict[str, Any],
    uid: int,
) -> SynapseStreamResult:
    
    try:
        with Timer() as timer:
            axon_info = metagraph.axons[uid]
            miner = openai.AsyncOpenAI(
                base_url=f"http://{axon_info.ip}:{axon_info.port}/v1", #Maybe need to change this? 
                api_key="Apex",
                max_retries=0,
                timeout=Timeout(settings.NEURON_TIMEOUT, connect=5, read=5),
                http_client=openai.DefaultAsyncHttpxClient(event_hooks={
                    "request": [
                        create_header_hook(
                            wallet.hotkey, axon_info.hotkey, task
                        )
                    ]
                }),
            )
            try:
                chunk_timings = []
                chunks = []
                chat = await miner.chat.completions.create(**generate_header(wallet.hotkey, body, signed_for=axon_info.hotkey))
                async for chunk in chat:
                    if chunk.choices[0].delta is None:
                        continue
                    if (
                        chunk.choices[0].delta.content == ""
                        or chunk.choices[0].delta.content is None
                    ) and len(chunks) == 0:
                        continue
                    
                    chunks.append(chunk.choices[0].delta.content)
                    chunk_timings.append(timer.elapsed_time())

            except openai.APIConnectionError as e:
                bt.logging.trace(f"Miner {uid} failed request: {e}")

            except Exception as e:
                bt.logging.trace(f"Unknown Error when sending to miner {uid}: {e}")

    except Exception as e:
        exception = e
        bt.logging.error(f"{uid}: Error in forward for: {e}")
        bt.logging.error(traceback.format_exc())
    finally:
        return SynapseStreamResult(
            accumulated_chunks=chunks,
            accumulated_chunks_timings=chunk_timings,
            synapse=None,
            uid=uid,
            exception=exception,
        )