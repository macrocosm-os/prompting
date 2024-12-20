import random

from fastapi import APIRouter, Request
from loguru import logger
from starlette.responses import StreamingResponse

from validator_api.chat_completion import chat_completion
from validator_api.mixture_of_miners import mixture_of_miners

router = APIRouter()


<<<<<<< HEAD
=======
async def forward_response(uid: int, body: dict[str, any], chunks: list[str]):
    uid = int(uid)  # sometimes uid is type np.uint64
    logger.info(f"Forwarding response to scoring with body: {body}")
    if not shared_settings.SCORE_ORGANICS:  # Allow disabling of scoring by default
        return

    if body.get("task") != "InferenceTask":
        logger.debug(f"Skipping forwarding for non-inference task: {body.get('task')}")
        return
    url = f"http://{shared_settings.VALIDATOR_API}/scoring"
    payload = {"body": body, "chunks": chunks, "uid": uid}
    try:
        timeout = httpx.Timeout(timeout=120.0, connect=60.0, read=30.0, write=30.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url, json=payload, headers={"api-key": shared_settings.SCORING_KEY, "Content-Type": "application/json"}
            )
            if response.status_code == 200:
                logger.info(f"Forwarding response completed with status {response.status_code}")

            else:
                logger.exception(
                    f"Forwarding response uid {uid} failed with status {response.status_code} and payload {payload}"
                )

    except Exception as e:
        logger.error(f"Tried to forward response to {url} with payload {payload}")
        logger.exception(f"Error while forwarding response: {e}")


>>>>>>> staging
@router.post("/v1/chat/completions")
async def completions(request: Request):
    """Main endpoint that handles both regular and mixture of miners chat completion."""
    try:
        body = await request.json()
        body["seed"] = int(body.get("seed") or random.randint(0, 1000000))

        # Choose between regular completion and mixture of miners.
        if body.get("mixture", False):
            return await mixture_of_miners(body)
        else:
<<<<<<< HEAD
            return await chat_completion(body)
=======
            logger.info("Forwarding response to scoring...")
            asyncio.create_task(forward_response(uid=uid, body=body, chunks=response[1]))
            return response[0]
>>>>>>> staging

    except Exception as e:
        logger.exception(f"Error in chat completion: {e}")
        return StreamingResponse(content="Internal Server Error", status_code=500)
