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


class Synapse(BaseModel):
    """Base class for all synapses"""


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

    timeout: int = 15
    metagraph: bt.metagraph

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
        # Prepare the request
        headers, raw_body = self.prepare_request(synapse)

        # Get the endpoint URL
        if miner_uids:
            urls = [url + f"/{synapse.__class__.__name__}" for url in await self.get_miner_urls(miner_uids)]
        else:
            urls = ["http://localhost:8000/streaming_synapse"]  # Default for testing

        logger.debug(f"Sending request to: {urls}")

        # Set timeout
        timeout = ClientTimeout(total=self.timeout)

        # Create session and send requests concurrently
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                self.send_single_request(session=session, url=url, headers=headers, raw_body=raw_body, uid=uid)
                for url, uid in zip(urls, miner_uids)
            ]

            # Gather all responses
            responses = await asyncio.gather(*tasks)

        return responses
