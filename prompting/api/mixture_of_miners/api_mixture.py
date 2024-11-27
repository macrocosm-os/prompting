from fastapi import APIRouter, Depends, Request

from prompting.api.api_managements.api import validate_api_key

router = APIRouter()


TASK_SYSTEM = {
    None: """You have been provided with a set of responses from various open-source models to the latest user query.
Your task is to synthesize these responses into a single, high-quality and concise response.
It is crucial to follow the provided instuctions or examples in the given prompt if any, and ensure the answer is in correct and expected format.
Critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.
Your response should not simply replicate the given answers but should offer a refined and accurate reply to the instruction.
Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
Responses from models:""",
}


@router.post("/mixture_of_miners")
async def mixture_of_miners(
    request: Request,
    api_key_data: dict = Depends(validate_api_key),
):
    return {"message": "Mixture of Miners"}
