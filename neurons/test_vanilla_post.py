import openai

from httpx import Timeout
from prompting import settings

settings.settings = settings.Settings.load(mode="validator")
settings = settings.settings

from prompting.base.epistula import create_header_hook

async def main():

    payload = {
        "seed": "42",
        "sampling_parameters": settings.SAMPLING_PARAMS,
        "task": "InferenceTask",
        "model": "Dummy_Model",
        "messages": [
            {"role": "user", "content": "#Bittensor #ToTheMoon"},
        ],
    }

    uid = 732
    try:
        axon_info = settings.METAGRAPH.axons[uid]
        miner = openai.AsyncOpenAI(
            base_url=f"http://{axon_info.ip}:{axon_info.port}/v1",
            api_key="Apex",
            max_retries=0,
            timeout=Timeout(settings.NEURON_TIMEOUT, connect=5, read=10),
            http_client=openai.DefaultAsyncHttpxClient(
                event_hooks={"request": [create_header_hook(settings.WALLET.hotkey, axon_info.hotkey)]}
            ),
        )
        chat = await miner.chat.completions.create(
            messages=payload["messages"],
            model=payload["model"],
            stream=True,
            extra_body={k: v for k, v in payload.items() if k not in ["messages", "model"]},
        )

        async for chunk in chat:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content)
    except Exception as e:
        print("something went wrong", e)
    return

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())