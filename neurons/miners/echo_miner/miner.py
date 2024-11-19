from fastapi import FastAPI, Request
import uvicorn

from prompting.base.epistula import (
    EpistulaRequest,
    verify_and_process_request,
)
from prompting.base.protocol import StreamPromptingSynapse

# Initialize FastAPI app
app = FastAPI()


@app.post("/StreamPromptingSynapse", response_model=StreamPromptingSynapse)
async def handle_request(request: Request, epistula_request: EpistulaRequest):
    """Handle incoming Epistula requests"""
    synapse: StreamPromptingSynapse = await verify_and_process_request(
        request, epistula_request, SynapseType=StreamPromptingSynapse
    )
    synapse.completion = "Echoing back: " + synapse.messages[-1]
    return synapse


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
