import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from loguru import logger

from prompting.llms.model_zoo import ModelZoo
from prompting.rewards.scoring import task_scorer
from prompting.tasks.inference import InferenceTask
from shared.base import DatasetEntry
from shared.dendrite import DendriteResponseEvent
from shared.epistula import SynapseStreamResult
from shared.settings import shared_settings

router = APIRouter()


def validate_scoring_key(api_key: str = Header(...)):
    if api_key != shared_settings.SCORING_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@router.post("/scoring")
async def score_response(request: Request, api_key_data: dict = Depends(validate_scoring_key)):
    model = None
    payload: dict[str, Any] = await request.json()
    body = payload.get("body")

    try:
        if body.get("model") is not None:
            model = ModelZoo.get_model_by_id(body.get("model"))
    except Exception:
        logger.warning(
            f"Organic request with model {body.get('model')} made but the model cannot be found in model zoo. Skipping scoring."
        )
        return
    uid = int(payload.get("uid"))
    chunks = payload.get("chunks")
    llm_model = ModelZoo.get_model_by_id(model) if (model := body.get("model")) else None
    task_scorer.add_to_queue(
        task=InferenceTask(
            messages=[msg["content"] for msg in body.get("messages")],
            llm_model=llm_model,
            llm_model_id=body.get("model"),
            seed=int(body.get("seed", 0)),
            sampling_params=body.get("sampling_params", {}),
        ),
        response=DendriteResponseEvent(
            uids=[uid],
            stream_results=[SynapseStreamResult(accumulated_chunks=[chunk for chunk in chunks if chunk is not None])],
            timeout=shared_settings.NEURON_TIMEOUT,
        ),
        dataset_entry=DatasetEntry(),
        block=shared_settings.METAGRAPH.block,
        step=-1,
        task_id=str(uuid.uuid4()),
    )
    logger.info("Organic tas appended to scoring queue")
