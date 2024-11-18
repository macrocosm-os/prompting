from fastapi import FastAPI
from prompting.api.GPTEndpoints.api import router as gpt_router

app = FastAPI()
app.include_router(gpt_router, prefix="/gpt", tags=["gpt"])
