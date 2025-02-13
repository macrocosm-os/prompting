import asyncio
import contextlib

import uvicorn
from fastapi import FastAPI

from shared import settings

settings.shared_settings = settings.SharedSettings.load(mode="api")
shared_settings = settings.shared_settings

from validator_api import scoring_queue
from validator_api.api_management import router as api_management_router
from validator_api.gpt_endpoints import router as gpt_router
from validator_api.utils import update_miner_availabilities_for_api


# Create the FastAPI app with the lifespan handler.
app = FastAPI()
app.include_router(gpt_router, tags=["GPT Endpoints"])
app.include_router(api_management_router, tags=["API Management"])


@app.get("/health")
async def health():
    return {"status": "ok"}


async def main():
    # Start background tasks.
    availability_task = asyncio.create_task(update_miner_availabilities_for_api.start())
    scoring_task = asyncio.create_task(scoring_queue.scoring_queue.start())

    # Configure Uvicorn.
    # Note: workers is set to 1 here so that the background tasks run in the same process.
    config = uvicorn.Config(
        app,
        host=shared_settings.API_HOST,
        port=shared_settings.API_PORT,
        log_level="debug",
        timeout_keep_alive=60,
        workers=shared_settings.WORKERS,
        reload=False,
    )
    server = uvicorn.Server(config)

    try:
        # Await the server. This will run concurrently with the background tasks.
        await server.serve()
    finally:
        # When the server shuts down, cancel the background tasks.
        availability_task.cancel()
        scoring_task.cancel()
        await asyncio.gather(availability_task, scoring_task, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
