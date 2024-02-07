# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import bittensor as bt
import argparse
import asyncio
from functools import partial
from starlette.types import Send
from typing import Awaitable, List


# Bittensor Miner Template:
from prompting.protocol import StreamPromptingSynapse

# import base miner class which takes care of most of the boilerplate
from neurons.miner import StreamMiner

from openai_utils import OpenAIUtils

from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage

class OpenAIStreamMiner(StreamMiner, OpenAIUtils):
    """Langchain-based miner which uses OpenAI's API as the LLM.

    You should also install the dependencies for this miner, which can be found in the requirements.txt file in this directory.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds OpenAI-specific arguments to the command line parser.
        """
        super().add_args(parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        bt.logging.info(f"Initializing with model {self.config.neuron.model_id}...")

        if self.config.wandb.on:
            self.identity_tags = ("openai_miner",) + (self.config.neuron.model_id,)

        _ = load_dotenv(find_dotenv())
        api_key = os.environ.get("OPENAI_API_KEY")

        self.async_callback = AsyncIteratorCallbackHandler()

        # Set openai key and other args
        self.model = ChatOpenAI(
            api_key=api_key,
            model_name=self.config.neuron.model_id,
            max_tokens=self.config.neuron.max_tokens,
            temperature=self.config.neuron.temperature,
            streaming=True,
            callbacks=[self.async_callback],
        )

        self.system_prompt = "You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know."
        self.accumulated_total_tokens = 0
        self.accumulated_prompt_tokens = 0
        self.accumulated_completion_tokens = 0
        self.accumulated_total_cost = 0

    async def forward(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        bt.logging.debug(
            "RUNNING THE FORWARD METHOD OF OPENAISTREAM MINER"
        )  # This shouldn't run.
        pass  # This is a placeholder for the forward method since it's mandatory to have.

    def prompt(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        def format_return(buffer: List, more_body: bool, send: Send):
            """Format return should eventually wrap the r dictionary in the starlette Send class."""
            joined_buffer = "".join(buffer)

            bt.logging.info(f"Joined buffer: {joined_buffer}")

            r = send(
                {
                    "type": "http.response.body",
                    "body": joined_buffer.encode("utf-8"),
                    "more_body": more_body,
                }
            )
            return r

        async def _prompt(message: str, send: Send) -> Awaitable:
            buffer = []
            BATCH_SIZE = 3
            bt.logging.debug("ENTERING _PROMPT IN OPENAISTREAM MINER")

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
                    fn=self.model.agenerate(messages=[[HumanMessage(content=message)]]),
                    event=self.async_callback.done,
                ),
            )

            async for token in self.async_callback.aiter():
                buffer.append(token)
                if len(buffer) == BATCH_SIZE:
                    print("Current buffer: ", buffer)
                    # r = format_return(self.buffer, more_body=True, send=send)

                    joined_buffer = "".join(buffer)

                    bt.logging.info(f"Joined buffer: {joined_buffer}")

                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": True,
                        }
                    )

                    # await r
                    buffer = []

            if buffer:
                # r = format_return(self.buffer, more_body=False, send=send)
                # await r
                joined_buffer = "".join(buffer)
                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": False,
                    }
                )

            # await task  # Tasks are Awaitables that represent the execution of a coroutine in the background.

        message = synapse.messages[0]
        bt.logging.debug(f"message in _prompt: {message}")
        token_streamer = partial(_prompt, message)
        bt.logging.debug(f"token streamer: {token_streamer}")

        return synapse.create_streaming_response(token_streamer)


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with OpenAIStreamMiner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)

            if miner.should_exit:
                bt.logging.warning("Ending miner...")
                break
