from typing import Any
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from shared.settings import shared_settings
import uuid

from prompting.api.api_managements.api import validate_api_key
from prompting.llms.model_zoo import ModelZoo
from prompting.rewards.scoring import task_scorer
from prompting.tasks.inference import InferenceTask
from shared.dendrite import DendriteResponseEvent
from shared.epistula import SynapseStreamResult
from shared.base import DatasetEntry

router = APIRouter()


# class ScoringPayload(BaseModel):
#     body: dict[str, Any]
#     chunks: list[str]
#     uid: int


@router.post("/scoring")
async def score_response(request: Request):  # , api_key_data: dict = Depends(validate_api_key)):
    payload: dict[str, Any] = await request.json()
    body = payload.get("body")
    uid = payload.get("uid")
    chunks = payload.get("chunks")
    task_scorer.add_to_queue(
        task=InferenceTask(
            messages=body.get("messages"),
            llm_model=ModelZoo.get_model_by_id(model) if (model := body.get("model")) else None,
            llm_model_id=body.get("model"),
            seed=int(body.get("seed")),
            sampling_params=body.get("sampling_parameters"),
        ),
        response=DendriteResponseEvent(
            uids=[uid],
            stream_results=[SynapseStreamResult(accumulated_chunks=chunks)],
            timeout=shared_settings.NEURON_TIMEOUT,
        ),
        dataset_entry=DatasetEntry(),
        block=-1,
        step=-1,
        task_id=str(uuid.uuid4()),
    )
