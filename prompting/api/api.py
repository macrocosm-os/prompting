import uvicorn
from fastapi import FastAPI
from loguru import logger

from prompting.api.api_managements.api import router as api_management_router
from prompting.api.miner_availabilities.api import router as miner_availabilities_router
from prompting.api.scoring.api import router as scoring_router
from shared.settings import shared_settings

app = FastAPI()

app.include_router(api_management_router, prefix="/api_management", tags=["api_management"])
app.include_router(miner_availabilities_router, prefix="/miner_availabilities", tags=["miner_availabilities"])
app.include_router(scoring_router, tags=["scoring"])


@app.get("/health")
def health():
    logger.info("Health endpoint accessed.")
    return {"status": "healthy"}


async def start_scoring_api():
    logger.info("Starting API...")
    uvicorn.run(
        "prompting.api.api:app", host="0.0.0.0", port=shared_settings.SCORING_API_PORT, loop="asyncio", reload=False
    )
