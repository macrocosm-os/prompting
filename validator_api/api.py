import asyncio

import uvicorn
from fastapi import FastAPI

from shared import settings

settings.shared_settings = settings.SharedSettings.load(mode="api")
shared_settings = settings.shared_settings

from validator_api import scoring_queue
from validator_api.api_management import router as api_management_router
from validator_api.gpt_endpoints import router as gpt_router
from validator_api.utils import update_miner_availabilities_for_api

app = FastAPI()
app.include_router(gpt_router, tags=["GPT Endpoints"])
app.include_router(api_management_router, tags=["API Management"])


@app.get("/health")
async def health():
    return {"status": "ok"}


async def main():
    asyncio.create_task(update_miner_availabilities_for_api.start())
    asyncio.create_task(scoring_queue.scoring_queue.start())
    uvicorn.run(
        app,
        host=shared_settings.API_HOST,
        port=shared_settings.API_PORT,
        log_level="debug",
        timeout_keep_alive=60,
    )


if __name__ == "__main__":
    asyncio.run(main())
