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


@router.post("/scoring")
async def score_response(request: Request):  #, api_key_data: dict = Depends(validate_api_key)):
    payload: dict[str, Any] = await request.json()
    body = payload.get("body")
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
