import uvicorn
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from loguru import logger


from prompting.settings import settings
from prompting import mutable_globals
from prompting.tasks.inference import InferenceTask
from prompting.rewards.scoring import ScoringConfig
from prompting.datasets.base import DatasetEntry
from shared.dendrite import DendriteResponseEvent, SynapseStreamResult

app = FastAPI()

def verify_api_key(authorization: str = Header(None)):
    """
    Dependency to verify API key from Authorization header.
    """
    if not authorization or not authorization.startswith("Bearer "):
        logger.error("Unauthorized: Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ")[1]
    if token != settings.SCORING_KEY:
        logger.error("Unauthorized: Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True

@app.post("/scoring")
async def scoring_endpoint(
    request: Request, 
    verified: bool = Depends(verify_api_key)
):
    try:
        payload = await request.json()
        logger.debug(f"Received scoring request: {request}")
        
        body = payload.get("body")
        if not body:
            logger.error("Bad Request: 'scoring_info' is missing from payload")
            raise HTTPException(status_code=400, detail="body (the original request) is required in payload")
        
        if body.get("task") != "InferenceTask":
            logger.warning("A non inference task has been received, which is not supported for organic scoring. Skipping...")
        else:
            try:
                # Create a task w/reference
                task = InferenceTask(query = body.get("messages")[-1].get("content"), seed = body.get("seed"), 
                                    sampling_params=body.get("sampling_parameters"), llm_model=body.get("model"))
                task.generate_reference()
                mutable_globals.scoring_queue.append(
                    ScoringConfig(
                        task = task, 
                        response = DendriteResponseEvent(
                            uids=payload.get("uid"),
                            timeout = settings.NEURON_TIMEOUT, 
                            stream_results=[
                                SynapseStreamResult(
                                    accumulated_chunks=payload.get("chunks"),
                                    accumulated_chunks_timings=[0 for chunk in payload.get("chunks")],
                                    uid=payload.get("uid"),
                                    exception=None,
                                    status_code=200 if payload.get("chunks") else 0,
                                    status_message="success" if payload.get("chunks") else "errored",
                                )
                            ],
                            #TODO: MAY HAVE TO ADD THE REST OF THE DENDRITE RESPONSE EVENT FIELDS
                        ), 
                        dataset_entry=DatasetEntry(),
                        block = settings.METAGRAPH.block,
                        step = 99999,
                        task_id="organic",
                    )
                )
            except Exception as e:
                logger.exception(f"Could not create scoring task for response: {e}. Skipping...")
        return
    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        raise HTTPException(status_code=400, detail="Invalid request payload")


async def start_scoring_endpoint():
    if not settings.SCORING_ENDPOINT_HOST and settings.SCORING_ENDPOINT_PORT:
        logger.exception("If you are deploying an api you must set SCORING_ENDPOINT_HOST and SCORING_ENDPOINT_PORT in the .env.validator")
        raise RuntimeError
    uvicorn.run(
        app,
        host=settings.SCORING_ENDPOINT_HOST,
        port=settings.SCORING_ENDPOINT_PORT,
        log_level="debug",
        timeout_keep_alive=60,
    )