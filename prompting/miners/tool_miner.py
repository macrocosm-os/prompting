import os
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
from traceback import print_exception
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
        # Message needs to be limited to 300 characters for wikipedia search, otherwise it will a return an error
        matches = wikipedia.search(message[:300])

        # If we find a match, we add the context to the system prompt
        if len(matches) > 0:
            title = matches[0]
            page = wikipedia.page(title)
            context = page.content

            if len(context) > 12_000:
                context = context[:12_000]

            return self.system_prompt.format(context=context)
        return self.config.neuron.system_prompt

    def forward(self, synapse: StreamPromptingSynapse):
        def format_send(buffer: List[str], more_body: bool):
            joined_buffer = "".join(buffer)
            bt.logging.debug(f"Streamed tokens: {joined_buffer}")
            return {
                "type": "http.response.body",
                "body": joined_buffer.encode("utf-8"),
                "more_body": more_body,
            }

        async def _forward(
            batch_size: int,
            chain: RunnableSequence,
            chain_formatter: Dict[str, str],
            send: Send,
        ):
            buffer = []

            # Langchain built in streaming. 'astream' also available for async
            for token in chain.stream(chain_formatter):
                buffer.append(token)

                if len(buffer) == batch_size:
                    await send(format_send(buffer, more_body=True))
                    buffer = []

            if buffer:
                await send(format_send(buffer, more_body=False))

        try:
            bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")

            role = synapse.roles[-1]
            message = synapse.messages[-1]

            formatted_system_prompt = self.format_system_prompt(message=message)

            prompt = ChatPromptTemplate.from_messages(
                [("system", formatted_system_prompt), ("user", "{input}")]
            )
            chain = prompt | self.model | StrOutputParser()
            chain_formatter = {"role": role, "input": message}

            token_streamer = partial(_forward, self.BATCH_SIZE, chain, chain_formatter)
            return synapse.create_streaming_response(token_streamer)

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            bt.logging.error(print_exception(value=e))
            synapse.completion = "Error: " + str(e)
        finally:
            if self.config.neuron.stop_on_forward_exception:
                self.should_exit = True
            return synapse

    async def blacklist(
        self, synapse: StreamPromptingSynapse
    ) -> typing.Tuple[bool, str]:
        return False, "All good here"

    async def priority(self, synapse: StreamPromptingSynapse) -> float:
        return 1e6
