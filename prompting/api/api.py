import uvicorn
from fastapi import FastAPI
from loguru import logger

from prompting.api.miner_availabilities.api import router as miner_availabilities_router
from prompting.api.scoring.api import router as scoring_router
from shared import settings

app = FastAPI()
app.include_router(miner_availabilities_router, prefix="/miner_availabilities", tags=["miner_availabilities"])
app.include_router(scoring_router, tags=["scoring"])


@app.get("/health")
def health():
    return {"status": "healthy"}


async def start_scoring_api(task_scorer, scoring_queue, reward_events):
    # We pass an object of task scorer then override it's attributes to ensure that they are managed by mp
    app.state.task_scorer = task_scorer
    app.state.task_scorer.scoring_queue = scoring_queue
    app.state.task_scorer.reward_events = reward_events

    logger.info(f"Starting Scoring API on https://0.0.0.0:{settings.shared_settings.SCORING_API_PORT}")
    uvicorn.run(
        "prompting.api.api:app",
        host="0.0.0.0",
        port=settings.shared_settings.SCORING_API_PORT,
        loop="asyncio",
        reload=False,
    )
