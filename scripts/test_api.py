import asyncio
import openai
from httpx import Timeout


async def make_completion(client: openai.AsyncOpenAI, prompt: str, stream: bool = False, seed: str = "1759348") -> str:
    """Make a completion request to the API.

    Args:
        miner: Configured AsyncOpenAI client
        prompt: Input prompt
        stream: Whether to stream the response
        seed: Random seed for reproducibility

    Returns:
        Generated completion text
    """
    result = await client.chat.completions.create(
        model=None,
        messages=[{"role": "user", "content": prompt}],
        stream=stream,
        extra_body={
            "seed": seed,
            "sampling_parameters": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
                "max_new_tokens": 256,
                "do_sample": True,
                "seed": None,
            },
            "task": "QuestionAnsweringTask",
            "mixture": False,
        },
    )

    if not stream:
        return result
    else:
        chunks = []
        async for chunk in result:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        return "".join(chunks)


async def main():
    PORT = 8005
    # Example API key, replace with yours:
    API_KEY = "0566dbe21ee33bba9419549716cd6f1f"
    client = openai.AsyncOpenAI(
        base_url=f"http://localhost:{PORT}/v1",
        max_retries=0,
        timeout=Timeout(90, connect=30, read=60),
        api_key=API_KEY,
    )
    response = await make_completion(client=client, prompt="Say 10 random numbers between 1 and 100", stream=True)
    print(response)



asyncio.run(main())
