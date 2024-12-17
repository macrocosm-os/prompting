import uvicorn
from fastapi import FastAPI, Request
from loguru import logger

from prompting import mutable_globals
from prompting.api.api_managements.api import router as api_management_router
from prompting.api.miner_availabilities.api import router as miner_availabilities_router
from prompting.datasets.base import DatasetEntry
from prompting.rewards.scoring import ScoringConfig
from prompting.settings import settings
from prompting.tasks.base_task import BaseTextTask
from shared.dendrite import DendriteResponseEvent, SynapseStreamResult

app = FastAPI()

app.include_router(api_management_router, prefix="/api_management", tags=["api_management"])
app.include_router(miner_availabilities_router, prefix="/miner_availabilities", tags=["miner_availabilities"])


@app.post("/score")
async def score(request: Request):
    """Endpoint to receive a response for scoring.

    Example request:
    {
        "response": "Miner's response to be scored.",
        "uid": 9999,
        "query": "What is the capital of France?",
        "seed": 1234,
        "llm_model_id": "some-llm-model",
        "sampling_params": {"temperature": 0.7},
        "timeout": 15,
        "block": 42,
        "step": 2,
        "task_id": "custom_task_id",
        "response_uid": 9999
    }
    """
    try:
        data = await request.json()

        # Mandatory fields
        response_str = data.get("response")
        if response_str is None:
            return {"status": "error", "message": "'response' field is required"}

        uid = data.get("uids")
        if uid is None:
            return {"status": "error", "message": "'uid' field is required"}

        # Optional fields with defaults.
        query = data.get("query", "Dummy query")
        seed = data.get("seed", 1234)
        llm_model_id = data.get("llm_model_id", "some-llm-model")
        sampling_params = data.get("sampling_params", {})
        dataset_entry_id = data.get("dataset_entry_id", "dummy_dataset_entry_id")

        # Construct the BaseTextTask.
        task = BaseTextTask(
            query=query,
            seed=seed,
            llm_model_id=llm_model_id,
            sampling_params=sampling_params,
            dataset_entry_id=dataset_entry_id
        )

        # Build the SynapseStreamResult.
        response_uid = data.get("response_uid", 9999)
        stream_result = SynapseStreamResult(
            exception=None,
            uid=response_uid,
            accumulated_chunks=[response_str],
            accumulated_chunks_timings=[],
            tokens_per_chunk=[]
        )

        # Construct DendriteResponseEvent.
        timeout = data.get("timeout", 10)
        response_event = DendriteResponseEvent(
            stream_results=[stream_result],
            uids=[uid],
            timeout=timeout
        )

        # Construct DatasetEntry if provided, else empty.
        dataset_entry = DatasetEntry()

        # Optional scoring config fields.
        block = data.get("block", 0)
        step = data.get("step", 0)
        task_id = data.get("task_id", "")

        # Add to scoring queue.
        mutable_globals.scoring_queue.append(
            ScoringConfig(
                task=task,
                response=response_event,
                dataset_entry=dataset_entry,
                block=block,
                step=step,
                task_id=task_id,
            )
        )
        return {"status": "scoring_queued", "task_id": task_id}

    except Exception as e:
        logger.exception("Error processing the string for scoring")
        return {"status": "error", "message": str(e)}


@app.get("/health")
def health():
    logger.info("Health endpoint accessed.")
    return {"status": "healthy"}


async def start_scoring_api():
    logger.info("Starting API...")
    uvicorn.run("prompting.api.api:app", host="0.0.0.0", port=settings.SCORING_API_PORT, loop="asyncio", reload=False)