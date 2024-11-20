from fastapi import FastAPI
import uvicorn
from prompting.api.gpt_endpoints.api import router as gpt_router
from prompting.api.miner_availabilities.api import router as miner_availabilities_router
from loguru import logger

app = FastAPI()

app.include_router(gpt_router)
app.include_router(miner_availabilities_router)


async def start_api():
    logger.info("Starting API")
    uvicorn.run(app, host="0.0.0.0", port=8000)
