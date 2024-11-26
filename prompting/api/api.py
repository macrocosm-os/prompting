from fastapi import FastAPI
from loguru import logger

# This ensures uvicorn is imported first
import uvicorn

# Now we can safely import the rest
from prompting.api.api_managements.api import router as api_management_router
from prompting.api.miner_availabilities.api import router as miner_availabilities_router
from prompting.api.gpt_endpoints.api import router as gpt_router

app = FastAPI()

# Add routers at the application level
app.include_router(api_management_router, prefix="/api_management", tags=["api_management"])
app.include_router(miner_availabilities_router, prefix="/miner_availabilities", tags=["miner_availabilities"])
app.include_router(gpt_router, tags=["gpt"])


@app.get("/health")
def health():
    logger.info("Health endpoint accessed.")
    return {"status": "healthy"}


# if __name__ == "__main__":
async def start_api():
    logger.info("Starting API...")
    uvicorn.run("prompting.api.api:app", host="0.0.0.0", port=8004, loop="asyncio", reload=False)
