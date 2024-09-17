# ruff: noqa: E402
import bittensor as bt
import asyncio
import time
import traceback
from typing import Awaitable
from transformers import AutoTokenizer
from tokenizers import Tokenizer

from prompting import settings

settings.settings = settings.Settings(mode="validator")
settings = settings.settings
from prompting.base.protocol import StreamPromptingSynapse
from prompting.base.dendrite import SynapseStreamResult


# NET_UID = 1
NET_UID = 61
# NET_UID = 170

# ACC = "OTF"
ACC = "MY_VALIDATOR"

if ACC == "OTF":
    NETWORK = "finney"
    WALLET = "opentensor"
    HOTKEY = "main"
elif ACC == "MY_VALIDATOR":
    NETWORK = "test"
    WALLET = "validator"
    HOTKEY = "validator_hotkey"
else:
    raise ValueError("Invalid account type")

# Wallet and Subnet Setup
wallet = bt.wallet(name=WALLET, hotkey=HOTKEY)
subtensor = bt.subtensor(network=NETWORK)
subnet = subtensor.metagraph(netuid=NET_UID)

hotkey = wallet.hotkey.ss58_address
my_uid = subnet.hotkeys.index(wallet.hotkey.ss58_address)
sub = bt.subtensor(NETWORK)
mg = sub.metagraph(NET_UID)

if hotkey not in mg.hotkeys:
    print(f"Hotkey {hotkey} deregistered")
else:
    print(f"Hotkey {hotkey} is registered. UID: {my_uid}.")

active_uids = subnet.uids.tolist()
print(f"My hotkey is active: {my_uid in active_uids}")
print(f"Is validator permit: {subnet.validator_permit[my_uid]}")


async def process_stream(uid: int, async_iterator: list[Awaitable], tokenizer: Tokenizer) -> SynapseStreamResult:
    """Process a single response asynchronously."""
    synapse = None
    exception = None
    accumulated_chunks = []
    accumulated_chunks_timings = []
    accumulated_tokens_per_chunk = []
    start_time = time.time()

    try:
        synapse = None
        for response in async_iterator:
            async for chunk in response:
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    accumulated_chunks.append(chunk)
                    accumulated_chunks_timings.append(time.time() - start_time)
                    tokens_in_chunk = len(tokenizer.tokenize(chunk))
                    accumulated_tokens_per_chunk.append(tokens_in_chunk)
                else:
                    synapse = chunk

            if synapse is None or not isinstance(synapse, StreamPromptingSynapse):
                raise ValueError(f"Something went wrong with miner uid {uid}, Synapse is not StreamPromptingSynapse.")
    except Exception as e:
        exception = e
        traceback_details = traceback.format_exc()
        bt.logging.error(f"Error in generating reference or handling responses for uid {uid}: {e}\n{traceback_details}")
        failed_synapse = StreamPromptingSynapse(roles=["user"], messages=["failure"], completion="")
        synapse = failed_synapse
    finally:
        print()  # Print newline at the end of stream
        return SynapseStreamResult(
            accumulated_chunks=accumulated_chunks,
            accumulated_chunks_timings=accumulated_chunks_timings,
            tokens_per_chunk=accumulated_tokens_per_chunk,
            synapse=synapse,
            uid=uid,
            exception=exception,
        )


async def handle_response(stream_results_dict: dict[int, Awaitable], tokenizer: Tokenizer) -> list[SynapseStreamResult]:
    tasks_with_uid = [(uid, stream_results_dict[uid]) for uid, _ in stream_results_dict.items()]
    process_stream_tasks = [process_stream(uid, resp, tokenizer) for uid, resp in tasks_with_uid]
    processed_stream_results = await asyncio.gather(*process_stream_tasks, return_exceptions=True)
    return processed_stream_results


async def chat_loop():
    dendrite = bt.dendrite(wallet=wallet)
    model_name = "casperhansen/llama-3-8b-instruct-awq"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    roles = []
    messages = []

    while True:
        prompt = input("User: ")
        if prompt.lower() in ["exit", "quit"]:
            break

        roles.append("user")
        messages.append(prompt)
        timeout = 15

        streams_responses = await dendrite(
            axons=[subnet.axons[my_uid]],
            synapse=StreamPromptingSynapse(
                roles=roles,
                messages=messages,
                task_name="InferenceTask",
            ),
            timeout=timeout,
            deserialize=False,
            streaming=True,
        )

        stream_results_dict = {my_uid: streams_responses}
        handle_stream_responses_task = asyncio.create_task(handle_response(stream_results_dict, tokenizer))

        try:
            stream_results = await asyncio.wait_for(handle_stream_responses_task, timeout=timeout)
        except asyncio.TimeoutError:
            print("\nTimeout: No response received within the time limit.")
            continue

        response = stream_results[0].synapse.completion
        roles.append("assistant")
        messages.append(response)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(chat_loop())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
