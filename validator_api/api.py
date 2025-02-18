import asyncio
import contextlib

from loguru import logger
import uvicorn
from fastapi import FastAPI

from shared import settings

# Load shared settings
settings.shared_settings = settings.SharedSettings.load(mode="api")
shared_settings = settings.shared_settings

from validator_api import scoring_queue
from validator_api.api_management import router as api_management_router
from validator_api.gpt_endpoints import router as gpt_router
from validator_api.utils import update_miner_availabilities_for_api
from multiprocessing import Lock

_lock = Lock()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background tasks for each worker.
    # Note: When running with multiple workers, these tasks will be started in every process.
    try:
        from validator_api import api_management
        with _lock:
            api_management._keys = api_management.load_api_keys()
        miner_task = asyncio.create_task(update_miner_availabilities_for_api.start())
        scoring_task = asyncio.create_task(scoring_queue.scoring_queue.start())
    except BaseException as e:
        logger.exception(f"Exception {e}")
    try:
        yield
    finally:
        miner_task.cancel()
        scoring_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await miner_task
            await scoring_task


app = FastAPI(lifespan=lifespan)
app.include_router(gpt_router, tags=["GPT Endpoints"])
app.include_router(api_management_router, tags=["API Management"])


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Run the app with multiple workers using uvicorn.run()
    uvicorn.run(
        "validator_api.api:app",  # Reference to the application
        host=shared_settings.API_HOST,
        port=shared_settings.API_PORT,
        log_level="debug",
        timeout_keep_alive=60,
        workers=shared_settings.WORKERS,  # This will spawn the specified number of worker processes
        reload=False,
    )
