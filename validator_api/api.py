import asyncio
import uvicorn
from fastapi import FastAPI

from shared import settings

# Load settings.
settings.shared_settings = settings.SharedSettings.load(mode="api")
shared_settings = settings.shared_settings

from validator_api.api_management import router as api_management_router
from validator_api.gpt_endpoints import router as gpt_router
from validator_api.utils import update_miner_availabilities_for_api

app = FastAPI()
app.include_router(gpt_router, tags=["GPT Endpoints"])
app.include_router(api_management_router, tags=["API Management"])

# Using app.state to store the background task.
@app.on_event("startup")
async def startup_event():
    app.state.background_task = asyncio.create_task(update_miner_availabilities_for_api.start())

@app.on_event("shutdown")
async def shutdown_event():
    app.state.background_task.cancel()
    try:
        await app.state.background_task
    except asyncio.CancelledError:
        pass

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "validator_api.api:app",  # Ensure this module path is correct.
        host=shared_settings.API_HOST,
        port=shared_settings.API_PORT,
        log_level="debug",
        timeout_keep_alive=60,
        workers=shared_settings.WORKERS,  # e.g. 8
        reload=False,
    )
