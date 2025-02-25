import asyncio
import datetime
from collections import deque
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel

from shared import settings
from shared.epistula import create_header_hook
from shared.loop_runner import AsyncLoopRunner
from validator_api.validator_forwarding import ValidatorRegistry

validator_registry = ValidatorRegistry()

shared_settings = settings.shared_settings


class ScoringPayload(BaseModel):
    payload: dict[str, Any]
    retries: int = 0


class ScoringQueue(AsyncLoopRunner):
    """Performs organic scoring every `interval` seconds."""

    interval: float = shared_settings.SCORING_RATE_LIMIT_SEC
    scoring_queue_threshold: int = shared_settings.SCORING_QUEUE_API_THRESHOLD
    max_scoring_retries: int = 3
    _scoring_lock = asyncio.Lock()
    _scoring_queue: deque[ScoringPayload] = deque()

    async def wait_for_next_execution(self, last_run_time) -> datetime.datetime:
        """If scoring queue is small, execute immediately, otherwise wait until the next execution time."""
        async with self._scoring_lock:
            if self.scoring_queue_threshold < self.size > 0:
                # If scoring queue is small and non-empty, score immediately.
                return datetime.datetime.now()

        return await super().wait_for_next_execution(last_run_time)

    async def run_step(self):
        """Perform organic scoring: pop queued payload, forward to the validator API."""
        logger.debug("Running scoring step")
        async with self._scoring_lock:
            if not self._scoring_queue:
                return

            scoring_payload = self._scoring_queue.popleft()
            payload = scoring_payload.payload
            uids = payload["uids"]
            logger.info(f"Received new organic for scoring, uids: {uids}")
        try:
            vali_uid, vali_axon, vali_hotkey = validator_registry.get_available_axon()
            # get_available_axon() will return None if it cannot find an available validator, which will fail to unpack
            url = f"http://{vali_axon}/scoring"
        except Exception as e:
            logger.exception(f"Could not find available validator scoring endpoint: {e}")
        try:
            # Verify payload is JSON serializable before sending
            try:
                import json
                json.dumps(payload)
            except TypeError as e:
                logger.error(f"Payload is not JSON serializable: {e}")
                # If we can't serialize the payload, skip this scoring item
                return
                
            timeout = httpx.Timeout(timeout=120.0, connect=60.0, read=30.0, write=30.0, pool=5.0)
            # Add required headers for signature verification

            async with httpx.AsyncClient(
                timeout=timeout,
                event_hooks={"request": [create_header_hook(shared_settings.WALLET.hotkey, vali_hotkey)]},
            ) as client:
                response = await client.post(
                    url=url,
                    json=payload,
                    # headers=headers,
                )
                validator_registry.update_validators(uid=vali_uid, response_code=response.status_code)
                if response.status_code != 200:
                    # Raise an exception so that the retry logic in the except block handles it.
                    raise Exception(f"Non-200 response: {response.status_code} for uids {uids}")
                logger.debug(f"Forwarding response completed with status {response.status_code}")
        except Exception as e:
            if scoring_payload.retries < self.max_scoring_retries:
                scoring_payload.retries += 1
                async with self._scoring_lock:
                    self._scoring_queue.appendleft(scoring_payload)
                logger.error(f"Tried to forward response to {url} for uids {uids}. Exception: {e}. Queued for retry")
            else:
                logger.exception(f"Error while forwarding response after {scoring_payload.retries} retries: {e}")

    async def append_response(
        self, uids: list[int], body: dict[str, Any], chunks: list[list[str]], timings: list[list[float]] | None = None
    ):
        if not shared_settings.SCORE_ORGANICS:
            return

        if body.get("task") != "InferenceTask" and body.get("task") != "WebRetrievalTask":
            # logger.debug(f"Skipping forwarding for non-inference/web retrieval task: {body.get('task')}")
            return

        # Recursively convert Pydantic models to dictionaries
        def make_json_serializable(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            elif hasattr(obj, "dict"):  # For older Pydantic versions
                return obj.dict()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj

        # Convert the entire body to ensure all nested objects are JSON serializable
        body_copy = make_json_serializable(body)

        uids = [int(u) for u in uids]
        chunk_dict = {str(u): c for u, c in zip(uids, chunks)}
        if timings:
            timing_dict = {str(u): t for u, t in zip(uids, timings)}
        else:
            timing_dict = {}
        payload = {"body": body_copy, "chunks": chunk_dict, "uids": uids, "timings": timing_dict}
        scoring_item = ScoringPayload(payload=payload)

        async with self._scoring_lock:
            self._scoring_queue.append(scoring_item)

    @property
    def size(self) -> int:
        return len(self._scoring_queue)

    def __len__(self) -> int:
        return self.size


# TODO: Leaving it as a global var to make less architecture changes, refactor as DI.
scoring_queue = ScoringQueue()
