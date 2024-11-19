import bittensor as bt
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from collections import OrderedDict
import time
from fastapi import Request
from loguru import logger
from typing import Type
import aiohttp
import asyncio
import numpy as np
import requests
from aiohttp.client import ClientTimeout
from pydantic import model_validator
from fastapi import HTTPException
from typing import Literal
from typing import AsyncGenerator


class Synapse(BaseModel):
    """Base class for all synapses"""


class StreamingSynapse(Synapse):
    """Base class for streaming synapses"""

    is_streaming: bool = True


class EpistulaRequest(BaseModel):
    """Epistula protocol request format"""

    data: Any
    nonce: int = Field(description="Unix timestamp of when request was sent")
    signed_by: str = Field(description="Hotkey of sender / signer")
    signed_for: Optional[str] = Field(default=None, description="Hotkey of intended receiver")
    version: int = Field(default=1)


class EpistulaResponse(BaseModel):
    """Response format"""

    data: Synapse


class StreamingEpistulaResponse(EpistulaResponse):
    """Streaming response format that yields chunks of data"""

    data: StreamingSynapse


def ordered_json(obj: Union[Dict, List, str, int, float, bool, None]) -> str:
    """Create a deterministically ordered JSON string"""
    if isinstance(obj, dict):
        return "{" + ",".join(f'"{k}":{ordered_json(v)}' for k, v in sorted(obj.items())) + "}"
    elif isinstance(obj, list):
        return "[" + ",".join(ordered_json(x) for x in obj) + "]"
    elif isinstance(obj, str):
        return f'"{obj}"'
    elif obj is None:
        return "null"
    else:
        return str(obj).lower()  # for bools and numbers


def generate_request_body(synapse: Synapse, sender_hotkey: str, receiver_hotkey: Optional[str] = None) -> OrderedDict:
    """Generate request body with deterministic ordering"""
    return OrderedDict(
        [
            ("data", OrderedDict(synapse.model_dump().items())),
            ("nonce", time.time_ns()),
            ("signed_by", sender_hotkey),
            ("signed_for", receiver_hotkey),
            ("version", 1),
        ]
    )


def generate_signature(wallet: bt.wallet, body: Union[str, Dict, OrderedDict]) -> str:
    """Generate signature for request body"""
    if not isinstance(body, str):
        body = ordered_json(body)
    return "0x" + wallet.hotkey.sign(body.encode()).hex()


async def verify_and_process_request(
    request: Request, epistula_request: EpistulaRequest, SynapseType: Type[Synapse]
) -> Synapse:
    headers = dict(request.headers)
    # Get the raw request body before Pydantic parsing
    raw_body = await request.body()
    raw_body_str = raw_body.decode()

    # Verify using raw request
    if not verify_raw_request(headers, raw_body_str, epistula_request.signed_by):
        logger.error(f"Invalid signature: {epistula_request}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    try:
        synapse = SynapseType(**epistula_request.data)
    except Exception as e:
        logger.error(f"Error parsing synapse: {e}")
        raise HTTPException(status_code=400, detail="Error parsing synapse")
    return synapse


def verify_raw_request(headers: dict[str, str], raw_body: str, sender_hotkey: str) -> bool:
    """Verify the signature using raw request body"""
    try:
        signature = headers.get("body-signature")
        if not signature or not signature.startswith("0x"):
            logger.error(f"Invalid signature - signature must start with 0x: {signature}")
            return False
        keys = bt.Keypair(ss58_address=sender_hotkey)
        signature_bytes = bytes.fromhex(signature[2:])
        logger.debug(f"Verifying signature on body: {raw_body}")
        return keys.verify(raw_body.encode(), signature_bytes)
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False


class EpistulaClient(BaseModel):
    """Client for making Epistula protocol requests"""

    wallet: bt.wallet
    timeout: int = 15

    class Config:
        arbitrary_types_allowed = True

    def prepare_request(self, synapse: Synapse, receiver_hotkey: Optional[str] = None) -> tuple[Dict[str, str], str]:
        """Prepare request headers and body"""
        # Generate body
        body = generate_request_body(
            synapse=synapse, sender_hotkey=self.wallet.hotkey.ss58_address, receiver_hotkey=receiver_hotkey
        )
        # Convert to consistent JSON format
        raw_body = ordered_json(body)
        # Generate signature
        signature = generate_signature(self.wallet, raw_body)
        # Create headers
        headers = {"Content-Type": "application/json", "Body-Signature": signature}
        return headers, raw_body


class MetagraphEpistulaClient(EpistulaClient):
    """Epistula client that can query miners using the metagraph"""

    metagraph: bt.metagraph
    mode: Literal["validator", "mock"] = "validator"

    @model_validator(mode="after")
    def refresh_metagraph(self):
        self._refresh_metagraph()
        return self

    def _refresh_metagraph(self):
        """Refresh the metagraph data"""
        try:
            logger.info(f"Metagraph refreshed with {len(self.metagraph.axons)} axons")
        except Exception as e:
            logger.error(f"Failed to refresh metagraph: {e}")
            raise

    async def get_miner_urls(self, uids: list[Union[int, str]]) -> list[str]:
        """Get miner's API URL from UID"""
        # Convert uid to int if string
        uids = np.array(uids).astype(int)

        if np.max(uids) >= len(self.metagraph.axons):
            raise ValueError(f"Invalid UID: {np.max(uids)}. Max UID is {len(self.metagraph.axons) - 1}")

        axons = np.array(self.metagraph.axons)[uids]
        return [f"http://{axon.ip}:{axon.port}/" for axon in axons]

    async def send_single_request(
        self, session: aiohttp.ClientSession, url: str, headers: Dict[str, str], raw_body: str, uid: int
    ) -> Dict:
        """Send a single request and handle its response"""
        try:
            async with session.post(url, headers=headers, data=raw_body) as response:
                response_data = await response.json()
                return {"uid": uid, "status": response.status, "data": response_data, "success": True}
        except Exception as e:
            logger.error(f"Request failed for UID {uid}: {e}")
            return {"uid": uid, "status": 500, "error": str(e), "success": False}

    async def send_request(
        self,
        synapse: Synapse,
        miner_uids: Optional[list[Union[int, str]]] = None,
    ) -> requests.Response:
        """
        Send request to a miner's API endpoint

        Args:
            synapse: The synapse object to send
            miner_uid: UID of the miner to query (if using metagraph)
            receiver_hotkey: Optional receiver hotkey
            endpoint: Optional custom endpoint URL (overrides miner_uid)

        Returns:
            Response from the miner's API
        """
        if self.mode == "mock":
            logger.warning("Running in mock mode. Sending to local endpoint.")
            urls = [f"http://localhost:8000/{synapse.__class__.__name__}"] * len(miner_uids)  # Default for testing

        # Get the endpoint URL
        elif miner_uids:
            urls = [url + f"/{synapse.__class__.__name__}" for url in await self.get_miner_urls(miner_uids)]
        else:
            raise ValueError("No miner UIDs provided")

        # Set timeout
        timeout = ClientTimeout(total=self.timeout)

        # Prepare the request
        headers, raw_body = self.prepare_request(synapse)

        # Create session and send requests concurrently
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                self.send_single_request(session=session, url=url, headers=headers, raw_body=raw_body, uid=uid)
                for url, uid in zip(urls, miner_uids)
            ]

            # Gather all responses
            responses = await asyncio.gather(*tasks)
        logger.debug(f"Queried: {urls}, {miner_uids}\nResponses: {responses}")
        all_synapses = []
        for response in responses:
            if response["status"] == 200:
                try:
                    all_synapses.append(synapse.__class__(**response["data"]))
                except Exception as e:
                    logger.exception(f"Couldn't parse response back into synapse: {e}")
            else:
                logger.error(f"Request failed for UID {response['uid']}: {response['error']}")
            all_synapses.append(synapse)
        return all_synapses


async def stream_verify_and_process_request(
    request: Request, epistula_request: EpistulaRequest, SynapseType: Type[StreamingSynapse]
) -> AsyncGenerator[str, None]:
    """Verify and process a streaming request"""
    headers = dict(request.headers)
    raw_body = await request.body()
    raw_body_str = raw_body.decode()

    # Verify using raw request first
    if not verify_raw_request(headers, raw_body_str, epistula_request.signed_by):
        logger.error(f"Invalid signature: {epistula_request}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        synapse = SynapseType(**epistula_request.data)
    except Exception as e:
        logger.error(f"Error parsing synapse: {e}")
        raise HTTPException(status_code=400, detail="Error parsing synapse")

    return synapse


class StreamingEpistulaClient(MetagraphEpistulaClient):
    """Client for making streaming Epistula protocol requests"""

    async def stream_response(self, response: aiohttp.ClientResponse, uid: int) -> AsyncGenerator[dict, None]:
        """Stream the response data with source UID"""
        async for chunk in response.content.iter_chunks():
            if chunk:
                yield {"uid": uid, "data": chunk[0].decode()}

    async def handle_miner_stream(
        self, session: aiohttp.ClientSession, url: str, headers: dict, raw_body: str, uid: int
    ) -> AsyncGenerator[dict, None]:
        """Handle streaming from a single miner"""
        try:
            async with session.post(url, headers=headers, data=raw_body) as response:
                if response.status == 200:
                    async for chunk in self.stream_response(response, uid):
                        yield chunk
                else:
                    logger.error(f"Request failed for UID {uid} on endpoint {url}: {response.status}")
                    yield {"uid": uid, "error": f"Request failed with status {response.status}"}
        except Exception as e:
            logger.error(f"Streaming request failed for UID {uid} on endpoint {url}: {e}")
            yield {"uid": uid, "error": str(e)}

    async def send_streaming_request(
        self,
        synapse: StreamingSynapse,
        miner_uids: Optional[list[Union[int, str]]] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Send concurrent streaming requests to multiple miners' API endpoints

        Args:
            synapse: The streaming synapse object to send
            miner_uids: UIDs of the miners to query

        Yields:
            Dictionaries containing:
                - uid: The miner's UID
                - data: The chunk of streaming data
                OR
                - error: Error message if the stream failed
        """
        if not miner_uids:
            raise ValueError("No miner UIDs provided")

        # Prepare the request
        headers, raw_body = self.prepare_request(synapse)
        timeout = ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Get all miner URLs
            if self.mode == "mock":
                logger.warning("Running in mock mode. Sending to local endpoint.")
                urls = ["http://localhost:8000"] * len(miner_uids)
            else:
                urls = await self.get_miner_urls(miner_uids)

            # Create queues for each miner's stream
            queues = {uid: asyncio.Queue() for uid in miner_uids}

            async def stream_to_queue(uid: int, url: str):
                """Stream from a miner to its queue"""
                try:
                    async for chunk in self.handle_miner_stream(
                        session=session,
                        url=f"{url}/{synapse.__class__.__name__}",
                        headers=headers,
                        raw_body=raw_body,
                        uid=uid,
                    ):
                        await queues[uid].put(chunk)
                except Exception as e:
                    logger.error(f"Error in stream_to_queue for UID {uid}: {e}")
                finally:
                    # Signal this stream is done
                    await queues[uid].put(None)

            # Start all streams
            tasks = [asyncio.create_task(stream_to_queue(uid, url)) for uid, url in zip(miner_uids, urls)]

            # Track active queues
            active_queues = set(queues.keys())

            # Yield chunks as they arrive
            while active_queues:
                # Wait for data from any queue
                done, pending = await asyncio.wait(
                    [asyncio.create_task(queues[uid].get()) for uid in active_queues],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    try:
                        chunk = task.result()
                        if chunk is None:
                            # This stream is done
                            finished_uid = next(uid for uid in active_queues if queues[uid].empty())
                            active_queues.remove(finished_uid)
                        else:
                            yield chunk
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")

            # Clean up
            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)
