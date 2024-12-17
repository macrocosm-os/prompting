from fastapi import APIRouter, Depends
from pydantic import BaseModel

from prompting.api.api_managements.api import validate_api_key
from prompting.llms.model_zoo import ModelZoo
from prompting.rewards.scoring import task_scorer
from prompting.tasks.inference import InferenceTask
from shared.dendrite import DendriteResponseEvent
from shared.epistula import SynapseStreamResult

router = APIRouter()


class ScoringPayload(BaseModel):
    body: dict
    response: list[dict]
    uid: int


@router.post("/scoring")
def score_response(payload: ScoringPayload, api_key_data: dict = Depends(validate_api_key)):
    task_scorer.add_to_queue(
        task=InferenceTask(
            messages=payload.body.get("messages"),
            llm_model=ModelZoo.get_model_by_id(payload.body.get("model")),
            llm_model_id=payload.body.get("model"),
            seed=payload.body.get("seed"),
            sampling_params=payload.body.get("sampling_params"),
        ),
        response=DendriteResponseEvent(
            uids=[payload.uid], stream_results=[SynapseStreamResult(accumulated_chunks=payload.response)]
        ),
    )
    pass
