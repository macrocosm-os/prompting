from typing import Any
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from prompting.api.api_managements.api import validate_api_key
from prompting.llms.model_zoo import ModelZoo
from prompting.rewards.scoring import task_scorer
from prompting.tasks.inference import InferenceTask
from shared.dendrite import DendriteResponseEvent
from shared.epistula import SynapseStreamResult

router = APIRouter()


# class ScoringPayload(BaseModel):
#     body: dict[str, Any]
#     chunks: list[str]
#     uid: int


@router.post("/scoring")
async def score_response(request: Request, api_key_data: dict = Depends(validate_api_key)):
    payload: dict[str, Any] = await request.json()
    body = payload.get("body")
    uid = payload.get("uid")
    chunks = payload.get("chunks")
    task_scorer.add_to_queue(
        task=InferenceTask(
            messages=body.get("messages"),
            llm_model=ModelZoo.get_model_by_id(body.get("model")),
            llm_model_id=body.get("model"),
            seed=body.get("seed"),
            sampling_params=body.get("sampling_params"),
        ),
        response=DendriteResponseEvent(
            uids=[uid], stream_results=[SynapseStreamResult(accumulated_chunks=chunks)]
        ),
    )
