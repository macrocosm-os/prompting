from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from typing import AsyncIterable, Awaitable, List

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from prompting.protocol import StreamPromptingSynapse

from starlette.types import Send
import asyncio

from functools import partial


def prompt(synapse: StreamPromptingSynapse):
    def format_return(buffer: List, more_body: bool):
        """Format return should eventually wrap the r dictionary in the starlette Send class."""
        joined_buffer = "".join(buffer)
        r = {
            "type": "http.response.body",
            "body": joined_buffer.encode("utf-8"),
            "more_body": more_body,
        }  # No more tokens to send
        return r

    async def _prompt(message: str, send: Send) -> Awaitable:
        callback = AsyncIteratorCallbackHandler()

        system_prompt = "You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know."
        model = ChatOpenAI(
            api_key=OPENAIKEY,
            model_name="gpt-3.5-turbo-1106",
            max_tokens=256,
            temperature=0.70,
            streaming=True,
            callbacks=[callback],
        )

        async def wrap_done(fn: Awaitable, event: asyncio.Event):
            """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
            try:
                await fn
            except Exception as e:
                # TODO: handle exception
                print(f"Caught exception: {e}")
            finally:
                # Signal the aiter to stop.
                event.set()

        # create_task schedules the execution of a coroutine as a background task.
        # asyncio creates a generator called "task"
        task = asyncio.create_task(
            wrap_done(
                model.agenerate(messages=[[HumanMessage(content=message)]]),
                callback.done,
            ),
        )

        buffer = []

        async for token in callback.aiter():
            # r = send({"type": "http.response.body","body":token,"more_body": True})  # No more tokens to send
            buffer.append(token)

            if len(buffer) == 3:
                print("Current buffer: ", buffer)
                r = format_return(buffer, more_body=True)
                yield r
                buffer = []

        if buffer:
            print("Final buffer: ", buffer)
            r = format_return(buffer, more_body=False)
            yield r

        await task  # . Tasks are Awaitables that represent the execution of a coroutine in the background.

    message = synapse.messages[0]
    bt.logging.trace(f"message in _prompt: {message}")
    token_streamer = partial(_prompt, message)
    bt.logging.trace(f"token streamer: {token_streamer}")
    return synapse.create_streaming_response(token_streamer)

    # token_streamer = partial(send_message, message=synapse.messages[0])
    # return synapse.create_streaming_response(token_streamer=token_streamer)


# # THIS WORKS
# async for response in send_message("What is the best way to learn a new language?"):
#     print("response:", response)


# # IDK HOW TO TEST THIS
# synapse = StreamPromptingSynapse(
#     roles=["user"], messages=["what is the captial of texas?"]
# )
# sr = version_2_main(synapse)
