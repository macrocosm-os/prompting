from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def score_response():
    # implementation goes here
    pass
