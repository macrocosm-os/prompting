from typing import Optional

import openai
from httpx import Timeout

from prompting import settings
from prompting.base.epistula import create_header_hook

settings.settings = settings.Settings.load(mode="validator")
settings = settings.settings


def setup_miner_client(
    port: int = 8004, api_key: str = "123456", hotkey: Optional[str] = None  # Default key from your api_keys.json
) -> openai.AsyncOpenAI:
    """
    Setup an authenticated OpenAI client for the miner.

    Args:
        port: Port number for the local server
        api_key: API key for authentication
        hotkey: Optional wallet hotkey

    Returns:
        Configured AsyncOpenAI client
    """

    # Create headers with both API key and hotkey
    async def combined_header_hook(request):
        # Add API key header
        request.headers["api-key"] = api_key
        # Add any additional headers from the original header hook
        if hotkey:
            original_hook = create_header_hook(hotkey, None)
            await original_hook(request)
        return request

    return openai.AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        max_retries=0,
        timeout=Timeout(60, connect=20, read=40),
        http_client=openai.DefaultAsyncHttpxClient(event_hooks={"request": [combined_header_hook]}),
    )


async def make_completion(miner: openai.AsyncOpenAI, prompt: str, stream: bool = False, seed: str = "1759348") -> str:
    """
    Make a completion request to the API.

    Args:
        miner: Configured AsyncOpenAI client
        prompt: Input prompt
        stream: Whether to stream the response
        seed: Random seed for reproducibility

    Returns:
        Generated completion text
    """
    result = await miner.chat.completions.create(
        model=None,
        messages=[{"role": "user", "content": prompt}],
        stream=stream,
        extra_body={
            "seed": seed,
            "sampling_parameters": settings.SAMPLING_PARAMS,
            "task": "QuestionAnsweringTask",
            "mixture": False,
        },
    )

    if not stream:
        return result
    else:
        print("In the else")
        chunks = []
        async for chunk in result:
            print(chunk)
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        return "".join(chunks)


async def main():
    PORT = 8004
    API_KEY = "0566dbe21ee33bba9419549716cd6f1f"
    miner = setup_miner_client(
        port=PORT, api_key=API_KEY, hotkey=settings.WALLET.hotkey if hasattr(settings, "WALLET") else None
    )
    response = await make_completion(miner=miner, prompt="Say 10 random numbers between 1 and 100", stream=True)
    print(["".join(res.accumulated_chunks) for res in response])


# Run the async main function
import asyncio

asyncio.run(main())
