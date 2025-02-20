import asyncio
import contextlib

import uvicorn
from fastapi import FastAPI
from loguru import logger

from shared import settings

settings.shared_settings = settings.SharedSettings.load(mode="api")
shared_settings = settings.shared_settings

from validator_api import scoring_queue
from validator_api.api_management import router as api_management_router
from validator_api.gpt_endpoints import router as gpt_router
from validator_api.utils import update_miner_availabilities_for_api


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    scoring_task = asyncio.create_task(scoring_queue.scoring_queue.start())
    miner_task = asyncio.create_task(update_miner_availabilities_for_api.start())
    yield
    miner_task.cancel()
    scoring_task.cancel()
    try:
        await miner_task
        await scoring_task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)
app.include_router(gpt_router, tags=["GPT Endpoints"])
app.include_router(api_management_router, tags=["API Management"])


@app.get("/health")
async def health():
    return {"status": "ok"}


async def main():
    logger.info(f"Starting API with {shared_settings.WORKERS} worker(s).")
    config = uvicorn.Config(
        "validator_api.api:app",
        host=shared_settings.API_HOST,
        port=shared_settings.API_PORT,
        log_level="debug",
        timeout_keep_alive=60,
        # Note: The `workers` parameter is typically only supported via the CLI.
        # When running programmatically with `server.serve()`, only a single worker will run.
        workers=shared_settings.WORKERS,
        reload=False,
    )
    server = uvicorn.Server(config)

    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
