# ruff: noqa: E402
from prompting import settings

settings.settings = settings.Settings.load(mode="miner")
settings = settings.settings

import time
import asyncio
import json
import httpx
import netaddr
import uvicorn
import requests
import traceback
import bittensor as bt
from starlette.responses import JSONResponse
from loguru import logger
from fastapi import APIRouter, Depends, FastAPI, Request, HTTPException
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse
from bittensor.core.extrinsics.serving import serve_extrinsic
from bittensor.core.axon import FastAPIThreadedServer
from prompting.base.epistula import verify_signature


MODEL_ID: str = "gpt-3.5-turbo"
NEURON_MAX_TOKENS: int = 256
NEURON_TEMPERATURE: float = 0.7
NEURON_TOP_K: int = 50
NEURON_TOP_P: float = 0.95
NEURON_STREAMING_BATCH_SIZE: int = 12
NEURON_STOP_ON_FORWARD_EXCEPTION: bool = False

SYSTEM_PROMPT = """You are a helpful agent that does it's best to answer all questions!"""


class OpenAIMiner():
    
    def __init__(self):
        self.should_exit = False
        self.client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        headers={
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
    )
        print("OpenAI Key: ", settings.OPENAI_API_KEY)

    async def format_openai_query(self, request: Request):
        data = await request.json()
        
        # Extract the required fields
        openai_request = {}
        for key in ["messages", "model", "stream"]:
            if key in data:
                openai_request[key] = data[key]
        openai_request["model"] = MODEL_ID
        
        return openai_request
    
    async def create_chat_completion(self, request: Request):
        bt.logging.info(
            "\u2713",
            f"Getting Chat Completion request from {request.headers.get('Epistula-Signed-By', '')[:8]}!",
        )
        req = self.client.build_request(
            "POST", "chat/completions", json = await self.format_openai_query(request)
        )
        r = await self.client.send(req, stream=True)
        return StreamingResponse(
            r.aiter_raw(), background=BackgroundTask(r.aclose), headers=r.headers
        )

    # async def create_chat_completion(self, request: Request):
    #     bt.logging.info(
    #         "\u2713",
    #         f"Getting Chat Completion request from {request.headers.get('Epistula-Signed-By', '')[:8]}!",
    #     )
        
    #     async def word_stream():
    #         words = "YOUR RESPONSE WOULD GO HERE".split()
    #         for word in words:
    #             # Simulate the OpenAI streaming response format
    #             data = {
    #                 "choices": [
    #                     {
    #                         "delta": {"content": word + ' '},
    #                         "index": 0,
    #                         "finish_reason": None
    #                     }
    #                 ]
    #             }
    #             yield f"data: {json.dumps(data)}\n\n"
    #             await asyncio.sleep(0.1)  # Simulate a delay between words
    #         # Indicate the end of the stream
    #         data = {
    #             "choices": [
    #                 {
    #                     "delta": {},
    #                     "index": 0,
    #                     "finish_reason": "stop"
    #                 }
    #             ]
    #         }
    #         yield f"data: {json.dumps(data)}\n\n"
    #         yield "data: [DONE]\n\n"
        
    #     return StreamingResponse(word_stream(), media_type='text/event-stream')

    async def check_availability(self, request: Request):
        print("Checking availability")
        data = await request.json()
        task_availabilities = data.get('task_availabilities', {})
        llm_model_availabilities = data.get('llm_model_availabilities', {})
        
        # Set all task availabilities to True
        task_response = {key: True for key in task_availabilities}
        
        # Set all model availabilities to False (openai will not be able to handle seeded inference)
        model_response = {key: False for key in llm_model_availabilities}
        
        response = {
            'task_availabilities': task_response,
            'llm_model_availabilities': model_response
        }
        
        return response
    
    async def verify_request(
        self,
        request: Request,
    ):
        # We do this as early as possible so that now has a lesser chance
        # of causing a stale request
        now = round(time.time() * 1000)

        # We need to check the signature of the body as bytes
        # But use some specific fields from the body
        signed_by = request.headers.get("Epistula-Signed-By")
        signed_for = request.headers.get("Epistula-Signed-For")
        if signed_for != settings.WALLET.hotkey.ss58_address:
            raise HTTPException(
                status_code=400, detail="Bad Request, message is not intended for self"
            )
        if signed_by not in settings.METAGRAPH.hotkeys:
            raise HTTPException(status_code=401, detail="Signer not in metagraph")

        uid = settings.METAGRAPH.hotkeys.index(signed_by)
        stake = settings.METAGRAPH.S[uid].item()
        if not settings.NETUID == 61 and stake < 10000:
            bt.logging.warning(
                f"Blacklisting request from {signed_by} [uid={uid}], not enough stake -- {stake}"
            )
            raise HTTPException(status_code=401, detail="Stake below minimum: {stake}")
        # TODO: Implement the optional epistula signature verification
        # # If anything is returned here, we can throw
        # body = await request.body()
        # err = verify_signature(
        #     request.headers.get("Epistula-Request-Signature"),
        #     body,
        #     request.headers.get("Epistula-Timestamp"),
        #     request.headers.get("Epistula-Uuid"),
        #     signed_for,
        #     signed_by,
        #     now,
        # )
        # if err:
        #     bt.logging.error(err)
        #     raise HTTPException(status_code=400, detail=err)

    def run(self):

        external_ip = None #settings.EXTERNAL_IP
        if not external_ip or external_ip == "[::]":
            try:
                external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
                netaddr.IPAddress(external_ip)
            except Exception:
                bt.logging.error("Failed to get external IP")

        bt.logging.info(
            f"Serving miner endpoint {external_ip}:{settings.AXON_PORT} on network: {settings.SUBTENSOR_NETWORK} with netuid: {settings.NETUID}"
        )

        serve_success = serve_extrinsic(
            subtensor=settings.SUBTENSOR,
            wallet=settings.WALLET,
            ip=external_ip,
            port=settings.AXON_PORT,
            protocol=4,
            netuid=settings.NETUID,
        )
        if not serve_success:
            bt.logging.error("Failed to serve endpoint")
            return

        # Start  starts the miner's endpoint, making it active on the network.
        # change the config in the axon
        app = FastAPI()
        router = APIRouter()
        router.add_api_route(
            "/v1/chat/completions",
            self.create_chat_completion,
            dependencies=[Depends(self.verify_request)],
            methods=["POST"],
        )
        router.add_api_route(
            "/availability",
            self.check_availability,
            methods=["POST"],
        )
        app.include_router(router)
        fast_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=settings.AXON_PORT,
            log_level="info",
            loop="asyncio",
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        bt.logging.info(f"Miner starting at block: {settings.SUBTENSOR.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                time.sleep(1)
        except Exception as e:
            bt.logging.error(str(e))
            bt.logging.error(traceback.format_exc())
        self.shutdown()


if __name__ == "__main__":
    miner = OpenAIMiner()
    miner.run()
    logger.warning("Ending miner...")