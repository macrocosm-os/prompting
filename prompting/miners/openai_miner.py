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
import bittensor as bt
import argparse
from starlette.types import Send
from functools import partial
from typing import Dict, List

# Bittensor Miner Template:
from prompting.base.prompting_miner import BaseStreamPromptingMiner
from prompting.protocol import StreamPromptingSynapse

# import base miner class which takes care of most of the boilerplate

from utils import OpenAIUtils

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.runnables.base import RunnableSequence
from langchain.callbacks import get_openai_callback


class OpenAIMiner(BaseStreamPromptingMiner, OpenAIUtils):
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

        # Set openai key and other args
        self.model = ChatOpenAI(
            api_key=api_key,
            model_name=self.config.neuron.model_id,
            max_tokens=self.config.neuron.max_tokens,
            temperature=self.config.neuron.temperature,
        )

        self.system_prompt = self.config.neuron.system_prompt
        self.accumulated_total_tokens = 0
        self.accumulated_prompt_tokens = 0
        self.accumulated_completion_tokens = 0
        self.accumulated_total_cost = 0

        self.BATCH_SIZE = 12 #Number of tokens to stream at a time.

    def forward(self, synapse : StreamPromptingSynapse):
        def format_send(buffer: List[str], more_body: bool): 
            joined_buffer = "".join(buffer)
            bt.logging.debug(f"Streamed tokens: {joined_buffer}")
            return {
                    "type": "http.response.body",
                    "body": joined_buffer.encode("utf-8"),
                    "more_body": more_body,
                }
        
        async def _forward(batch_size: int, chain: RunnableSequence, chain_formatter: Dict[str,str], send:Send):
            buffer = [] 

            #Langchain built in streaming. 'astream' also available for async
            for token in chain.stream(chain_formatter):
                buffer.append(token) 

                if len(buffer) == batch_size: 
                    await send(format_send(buffer, more_body = True))
                    buffer = []

            if buffer:
                await send(format_send(buffer, more_body=False))


        prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("user", "{input}")]
        )
        chain = prompt | self.model | StrOutputParser()

        role = synapse.roles[-1]
        message = synapse.messages[-1]

        chain_formatter = {"role": role, "input": message}

        token_streamer = partial(_forward, self.batch_size, chain, chain_formatter)
        return synapse.create_streaming_response(token_streamer)
