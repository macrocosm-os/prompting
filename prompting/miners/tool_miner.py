import os
import time
import typing
import argparse
import bittensor as bt
import wikipedia
from functools import partial
from dotenv import load_dotenv, find_dotenv
from typing import Dict, List

from starlette.types import Send


# Bittensor Miner Template:
from prompting.protocol import StreamPromptingSynapse

# import base miner class which takes care of most of the boilerplate
from prompting.base.prompting_miner import BaseStreamPromptingMiner
from prompting.miners.utils import OpenAIUtils

from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence


class ToolMiner(BaseStreamPromptingMiner, OpenAIUtils):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        bt.logging.info(f"Initializing with model {self.config.neuron.model_id}...")

        if self.config.wandb.on:
            self.identity_tags = ("openai_miner",) + (self.config.neuron.model_id,)

        _ = load_dotenv(find_dotenv())
        api_key = os.environ.get("OPENAI_API_KEY")

        # Set openai key and other args
        self.model = ChatOpenAI(
            api_key=api_key,
            model_name=self.config.neuron.model_id,
            max_tokens=self.config.neuron.max_tokens,
            temperature=self.config.neuron.temperature,
        )

        self.system_prompt = """You are a nice AI assistant that uses the provided context to answer user queries.
        ## Context
        {context}
        """

    def format_system_prompt(self, message: str) -> str:
        bt.logging.debug(f"💬 Searching for wikipedia context...")
        # Message needs to be limited to 300 characters for wikipedia search, otherwise it will a return an error
        matches = wikipedia.search(message[:300])

        # If we find a match, we add the context to the system prompt
        if len(matches) > 0:
            title = matches[0]
            page = wikipedia.page(title)
            context = page.content

            if len(context) > 12_000:
                context = context[:12_000]

            bt.logging.debug(f"💬 Wiki context found: {context}")

            return self.system_prompt.format(context=context)

        bt.logging.debug(f"❌ No Wiki context found")
        return self.config.neuron.system_prompt

    def forward(self, synapse: StreamPromptingSynapse):
        async def _forward(
            self, 
            init_time: float,
            timeout_threshold: float,
            chain: RunnableSequence,
            chain_formatter: Dict[str, str],
            send: Send,
        ):
            buffer = []
            timeout_reached = False

            try:
                # Langchain built in streaming. 'astream' also available for async
                for token in chain.stream(chain_formatter):
                    buffer.append(token)

                    if time.time() - init_time > timeout_threshold:
                        bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
                        timeout_reached = True
                        break

                    if len(buffer) == self.config.neuron.streaming_batch_size:
                        joined_buffer = "".join(buffer)
                        bt.logging.debug(f"Streamed tokens: {joined_buffer}")
                        await send(
                            {
                                "type": "http.response.body",
                                "body": joined_buffer.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                        buffer = []

                if buffer and not timeout_reached: # Don't send the last buffer of data if timeout.
                    joined_buffer = "".join(buffer)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": False,
                        }
                    )

            except Exception as e:
                bt.logging.error(f"Error in forward: {e}")

            finally:
                if self.config.neuron.stop_on_forward_exception:
                    self.should_exit = True


        bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")

        timeout_threshold = synapse.timeout

        role = synapse.roles[-1]
        message = synapse.messages[-1]

        formatted_system_prompt = self.format_system_prompt(message=message)

        prompt = ChatPromptTemplate.from_messages(
            [("system", formatted_system_prompt), ("user", "{input}")]
        )
        chain = prompt | self.model | StrOutputParser()
        chain_formatter = {"role": role, "input": message}

        init_time = time.time()

        token_streamer = partial(
            _forward,
            self, 
            init_time,
            timeout_threshold,
            chain,
            chain_formatter,
        )
        return synapse.create_streaming_response(token_streamer)

    async def blacklist(
        self, synapse: StreamPromptingSynapse
    ) -> typing.Tuple[bool, str]:
        return False, "All good here"

    async def priority(self, synapse: StreamPromptingSynapse) -> float:
        return 1e6
