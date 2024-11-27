# ruff: noqa: E402
from prompting import settings

settings.settings = settings.Settings.load(mode="miner")
settings = settings.settings

import asyncio
import json
import time

import httpx
import netaddr
import requests
import uvicorn
from bittensor.core.axon import FastAPIThreadedServer
from bittensor.core.extrinsics.serving import serve_extrinsic
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from loguru import logger
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from prompting.base.epistula import verify_signature
from prompting.llms.hf_llm import ReproducibleHF

MODEL_ID: str = "gpt-3.5-turbo"
NEURON_MAX_TOKENS: int = 256
NEURON_TEMPERATURE: float = 0.7
NEURON_TOP_K: int = 50
NEURON_TOP_P: float = 0.95
NEURON_STREAMING_BATCH_SIZE: int = 12
NEURON_STOP_ON_FORWARD_EXCEPTION: bool = False
SHOULD_SERVE_LLM: bool = False
LOCAL_MODEL_ID = "casperhansen/llama-3-8b-instruct-awq"


class OpenAIMiner:
    def __init__(self):
        self.should_exit = False
        self.client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        if SHOULD_SERVE_LLM:
            self.llm = ReproducibleHF(model_id=LOCAL_MODEL_ID)
        else:
            self.llm = None

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
        if self.llm and request.headers.get("task", None) == "inference":
            return await self.create_inference_completion(request)
        req = self.client.build_request("POST", "chat/completions", json=await self.format_openai_query(request))
        r = await self.client.send(req, stream=True)
        return StreamingResponse(r.aiter_raw(), background=BackgroundTask(r.aclose), headers=r.headers)

    async def create_inference_completion(self, request: Request):
        async def word_stream():
            inference = await self.run_inference(request)
            words = inference.split()
            for word in words:
                # Simulate the OpenAI streaming response format
                data = {"choices": [{"delta": {"content": word + " "}, "index": 0, "finish_reason": None}]}
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.1)  # Simulate a delay between words
            # Indicate the end of the stream
            data = {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(word_stream(), media_type="text/event-stream")

    async def check_availability(self, request: Request):
        print("Checking availability")
        data = await request.json()
        task_availabilities = data.get("task_availabilities", {})
        llm_model_availabilities = data.get("llm_model_availabilities", {})

        # Set all task availabilities to True
        task_response = {key: True for key in task_availabilities}

        # Set all model availabilities to False (openai will not be able to handle seeded inference)
        model_response = {key: key == LOCAL_MODEL_ID for key in llm_model_availabilities}

        response = {"task_availabilities": task_response, "llm_model_availabilities": model_response}

        return response

    async def verify_request(
        self,
        request: Request,
    ):
        now = round(time.time() * 1000)

        signed_by = request.headers.get("Epistula-Signed-By")
        signed_for = request.headers.get("Epistula-Signed-For")
        if signed_for != settings.WALLET.hotkey.ss58_address:
            raise HTTPException(status_code=400, detail="Bad Request, message is not intended for self")
        if signed_by not in settings.METAGRAPH.hotkeys:
            raise HTTPException(status_code=401, detail="Signer not in metagraph")

        uid = settings.METAGRAPH.hotkeys.index(signed_by)
        stake = settings.METAGRAPH.S[uid].item()
        if not settings.NETUID == 61 and stake < 10000:
            logger.warning(f"Blacklisting request from {signed_by} [uid={uid}], not enough stake -- {stake}")
            raise HTTPException(status_code=401, detail="Stake below minimum: {stake}")

        body = await request.body()
        err = verify_signature(
            request.headers.get("Epistula-Request-Signature"),
            body,
            request.headers.get("Epistula-Timestamp"),
            request.headers.get("Epistula-Uuid"),
            signed_for,
            signed_by,
            now,
        )
        if err:
            logger.error(err)
            raise HTTPException(status_code=400, detail=err)

    def run(self):
        external_ip = None  # settings.EXTERNAL_IP
        if not external_ip or external_ip == "[::]":
            try:
                external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
                netaddr.IPAddress(external_ip)
            except Exception:
                logger.error("Failed to get external IP")

        logger.info(
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
            logger.error("Failed to serve endpoint")
            return

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
            workers=4,
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        logger.info(f"Miner starting at block: {settings.SUBTENSOR.block}")

        # Main execution loop.
        try:
            while not self.should_exit:
                time.sleep(1)
        except Exception as e:
            logger.error(str(e))
        self.shutdown()

    async def run_inference(self, request: Request) -> str:
        data = await request.json()
        try:
            response = self.llm.generate(
                data.get("messages"), sampling_params=data.get("sampling_parameters"), seed=data.get("seed")
            )
            return response
        except Exception as e:
            logger.error(f"An error occurred during text generation: {e}")
            return str(e)


if __name__ == "__main__":
    miner = OpenAIMiner()
    miner.run()
    logger.warning("Ending miner...")
