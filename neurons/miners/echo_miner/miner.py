from fastapi import FastAPI, Request
import uvicorn
from prompting import settings
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from typing import Callable
import traceback
import json
from loguru import logger
from fastapi.responses import StreamingResponse

settings.settings = settings.Settings.load(mode="miner")

from prompting.base.epistula import (  # noqa: E402
    EpistulaRequest,
    verify_and_process_request,
    stream_verify_and_process_request,
)
from prompting.base.protocol import StreamPromptingSynapse  # noqa: E402
from prompting.miner_availability.miner_availability import AvailabilitySynapse  # noqa: E402


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        try:
            # Try to read and log the request body
            body = await request.body()
            try:
                body_str = body.decode()
                body_json = json.loads(body_str)
                logger.debug(f"Request Body: {json.dumps(body_json, indent=2)}")
            except Exception as _:
                logger.debug(f"Raw Request Body: {body}")

            # Log headers
            logger.debug(f"Request Headers: {dict(request.headers)}")

            # Store body for later use
            await request.body()  # Consume body
            setattr(request.state, "_body", body)  # Store for reuse

            response = await call_next(request)
            return response

        except Exception as e:
            logger.error(f"Middleware error: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"detail": "Internal server error in middleware"})


app = FastAPI()

# Add the logging middleware
app.add_middleware(LoggingMiddleware)

# @app.post("/StreamPromptingSynapse", response_model=StreamPromptingSynapse)
# async def stream_prompting_synapse(request: Request, epistula_request: EpistulaRequest):
#     """Handle incoming Epistula requests"""
#     synapse: StreamPromptingSynapse = await verify_and_process_request(
#         request, epistula_request, SynapseType=StreamPromptingSynapse
#     )
#     synapse.completion = "Echoing back: " + synapse.messages[-1]
#     return synapse


@app.post("/AvailabilitySynapse", response_model=AvailabilitySynapse)
async def availability_synapse(request: Request, epistula_request: EpistulaRequest):
    """Handle incoming Epistula requests"""
    synapse: AvailabilitySynapse = await verify_and_process_request(
        request, epistula_request, SynapseType=AvailabilitySynapse
    )
    return synapse


@app.post("/StreamPromptingSynapse", response_model=StreamPromptingSynapse)
async def streaming_endpoint(
    request: Request,
    epistula_request: EpistulaRequest,
) -> StreamingResponse:
    """Example streaming endpoint implementation"""
    synapse = await stream_verify_and_process_request(request, epistula_request, StreamPromptingSynapse)

    async def generate():
        # Your streaming logic here
        for word in ["Echoing back: "] + synapse.messages[-1].split():
            yield word
            await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
