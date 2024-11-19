from fastapi import FastAPI, Request
import uvicorn
from prompting import settings

settings.settings = settings.Settings.load(mode="miner")

from prompting.base.epistula import (  # noqa: E402
    EpistulaRequest,
    verify_and_process_request,
)
from prompting.base.protocol import StreamPromptingSynapse  # noqa: E402
from prompting.miner_availability.miner_availability import AvailabilitySynapse  # noqa: E402

app = FastAPI()


@app.post("/StreamPromptingSynapse", response_model=StreamPromptingSynapse)
async def stream_prompting_synapse(request: Request, epistula_request: EpistulaRequest):
    """Handle incoming Epistula requests"""
    synapse: StreamPromptingSynapse = await verify_and_process_request(
        request, epistula_request, SynapseType=StreamPromptingSynapse
    )
    synapse.completion = "Echoing back: " + synapse.messages[-1]
    return synapse


@app.post("/AvailabilitySynapse", response_model=AvailabilitySynapse)
async def availability_synapse(request: Request, epistula_request: EpistulaRequest):
    """Handle incoming Epistula requests"""
    synapse: AvailabilitySynapse = await verify_and_process_request(
        request, epistula_request, SynapseType=AvailabilitySynapse
    )
    return synapse


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
