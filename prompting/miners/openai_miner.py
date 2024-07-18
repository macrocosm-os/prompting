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

import time
import os
import bittensor as bt
import argparse
from starlette.types import Send
from functools import partial
from typing import Dict, Awaitable

# Bittensor Miner Template:
from prompting.base.prompting_miner import BaseStreamPromptingMiner
from prompting.protocol import StreamPromptingSynapse

# import base miner class which takes care of most of the boilerplate

from prompting.miners.utils import OpenAIUtils
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from typing import List, Dict
from traceback import print_exception

# Define the type for a list of dictionaries


class OpenAIMiner(BaseStreamPromptingMiner, OpenAIUtils):
    """Langchain-based miner which uses OpenAI's API as the LLM.
    This miner does not use any tools or external APIs when processing requests - it relies entirely on the models' own representation and world model. In some cases, this can produce lower quality results.
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
        self.model = OpenAI(api_key=api_key)
                        
        self.system_prompt = self.config.neuron.system_prompt
        self.accumulated_total_tokens = 0
        self.accumulated_prompt_tokens = 0
        self.accumulated_completion_tokens = 0
        self.accumulated_total_cost = 0

    def forward(self, synapse: StreamPromptingSynapse) -> Awaitable:
        async def _forward(
            self,
            synapse: StreamPromptingSynapse,
            init_time: float,
            timeout_threshold: float,
            send: Send,
        ):
            buffer = []
            accumulated_chunks = []
            accumulated_chunks_timings = []
            messages = []
            temp_completion = ""  # for wandb logging
            timeout_reached = False
            

            try:                
                system_prompt_message = [{ 'role': 'system', 'content': self.system_prompt }]
                synapse_messages = [{'role': role, 'content': message} for role, message in zip(synapse.roles, synapse.messages)]
                
                messages = system_prompt_message + synapse_messages
                
                start_time = time.time()
                stream_response = self.model.chat.completions.create(
                    model=self.config.neuron.model_id,
                    messages=messages,
                    temperature=self.config.neuron.temperature,
                    max_tokens=self.config.neuron.max_tokens,
                    stream=True
                )
                                
                for chunk in stream_response:
                    chunk_content = chunk.choices[0].delta.content
                    
                    if chunk_content is None:
                        bt.logging.info("OpenAI returned chunk content with None")
                        continue
                    
                    accumulated_chunks.append(chunk_content)
                    accumulated_chunks_timings.append(time.time() - start_time)                          
                    
                    buffer.append(chunk_content)                                        

                    if time.time() - init_time > timeout_threshold:
                        bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
                        timeout_reached = True
                        break

                    if len(buffer) == self.config.neuron.streaming_batch_size:
                        joined_buffer = "".join(buffer)
                        temp_completion += joined_buffer
                        bt.logging.debug(f"Streamed tokens: {joined_buffer}")

                        await send(
                            {
                                "type": "http.response.body",
                                "body": joined_buffer.encode("utf-8"),
                                "more_body": True,
                            }
                        )
                        buffer = []

                if (
                    buffer and not timeout_reached
                ):  # Don't send the last buffer of data if timeout.
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
                bt.logging.error(print_exception(type(e), e, e.__traceback__))
                if self.config.neuron.stop_on_forward_exception:
                    self.should_exit = True

            finally:
                synapse_latency = time.time() - init_time
                if self.config.wandb.on:
                    self.log_event(
                        synapse=synapse,
                        timing=synapse_latency,
                        messages=messages,
                        accumulated_chunks=accumulated_chunks,
                        accumulated_chunks_timings = accumulated_chunks_timings,
                    )

        bt.logging.debug(f"📧 Message received from {synapse.dendrite.hotkey}, IP: {synapse.dendrite.ip}; \nForwarding synapse: {synapse}")
        
        init_time = time.time()
        timeout_threshold = synapse.timeout

        token_streamer = partial(
            _forward,
            self,
            synapse,
            init_time,
            timeout_threshold,            
        )
        return synapse.create_streaming_response(token_streamer)
