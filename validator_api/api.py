import argparse

import uvicorn
from fastapi import FastAPI

from shared import settings

settings.shared_settings = settings.SharedSettings.load(mode="api")
shared_settings = settings.shared_settings

from validator_api.api_management import router as api_management_router
from validator_api.gpt_endpoints import router as gpt_router

app = FastAPI()
app.include_router(gpt_router, tags=["GPT Endpoints"])
app.include_router(api_management_router, tags=["API Management"])


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the validator_api FastAPI server.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes to run (default: 4).")
    args = parser.parse_args()

    uvicorn.run(
        "validator_api.api:app",
        host=shared_settings.API_HOST,
        port=shared_settings.API_PORT,
        log_level="debug",
        timeout_keep_alive=60,
        workers=args.workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
