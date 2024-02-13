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

import torch
import argparse
import bittensor as bt
from functools import partial
from threading import Thread
from starlette.types import Send

# Bittensor Miner Template:
from prompting.protocol import StreamPromptingSynapse
from prompting.llm import load_pipeline
from prompting.llm import HuggingFaceLLM

# import base miner class which takes care of most of the boilerplate
from prompting.base.prompting_miner import BaseStreamPromptingMiner
from transformers import TextIteratorStreamer


class HuggingFaceMiner(BaseStreamPromptingMiner):
    """
    Base 🤗 Hugging Face miner, integrated with hf pipeline.
    To run this miner from the project root directory:

    python neurons/miners/huggingface/miner.py --wallet.name <wallet_name> --wallet.hotkey <wallet_hotkey> --neuron.model_id <model_id> --subtensor.network <network> --netuid <netuid> --axon.port <port> --axon.external_port <port> --logging.debug True --neuron.model_id HuggingFaceH4/zephyr-7b-beta --neuron.system_prompt "Hello, I am a chatbot. I am here to help you with your questions." --neuron.max_tokens 64 --neuron.do_sample True --neuron.temperature 0.9 --neuron.top_k 50 --neuron.top_p 0.95 --wandb.on True --wandb.entity sn1 --wandb.project_name miners_experiments
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds arguments to the command line parser.
        """
        super().add_args(parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        model_kwargs = None
        if self.config.neuron.load_in_8bit:
            bt.logging.info("Loading 8 bit quantized model...")
            model_kwargs = dict(
                torch_dtype=torch.float16,
                load_in_8bit=True,
            )

        if self.config.neuron.load_in_4bit:
            bt.logging.info("Loading 4 bit quantized model...")
            model_kwargs = dict(
                torch_dtype=torch.float32,
                load_in_4bit=True,
            )

        if self.config.wandb.on:
            self.identity_tags = ("hf_miner",)

            if self.config.neuron.load_in_8bit:
                self.identity_tags += ("8bit_quantization",)
            elif self.config.neuron.load_in_4bit:
                self.identity_tags += ("4bit_quantization",)

        self.BATCH_SIZE = 12  # Number of tokens to stream at a time.

        # Forces model loading behaviour over mock flag
        mock = (
            False if self.config.neuron.should_force_model_loading else self.config.mock
        )

        self.llm_pipeline, self.streamer = load_pipeline(
            model_id=self.config.neuron.model_id,
            device=self.device,
            mock=mock,
            model_kwargs=model_kwargs,
        )

        self.model_id = self.config.neuron.model_id
        self.system_prompt = self.config.neuron.system_prompt

    def forward(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        async def _forward(batch_size: int, streamer: TextIteratorStreamer, send: Send):
            """
            TextIteratorStreamer: stores print-ready text in a queue, to be used by a downstream application as an iterator.
            """
            try:
                bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")

                buffer = []

                for token in streamer:
                    buffer.append(token)

                    if len(buffer) == batch_size:
                        joined_buffer = "".join(buffer)

                        bt.logging.debug(f"Streamed tokens: {joined_buffer}")

                        await send(
                            {
                                "type": "http.response.body",
                                "body": joined_buffer.encode("utf-8"),
                                "more_body": True,
                            }
                        )

                        buffer = []  # Clearing the buffer.

                if buffer:
                    joined_buffer = "".join(buffer)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": False,
                        }
                    )
                    bt.logging.debug(f"Streamed tokens: {joined_buffer}")

                torch.cuda.empty_cache()

            except Exception as e:
                bt.logging.error(f"Error: {e}")
                synapse.completion = "Error: " + str(e)
            finally:
                if self.config.neuron.stop_on_forward_exception:
                    self.should_exit = True
                return synapse

        prompt = synapse.messages[-1]

        # TODO: is there a way to close this thread? Not sure if this is hanging or not.
        # Create an async thread to generate the data in parallel to the streamer.
        thread = Thread(
            target=HuggingFaceLLM(
                llm_pipeline=self.llm_pipeline,
                system_prompt=self.system_prompt,
                max_new_tokens=self.config.neuron.max_tokens,
                do_sample=self.config.neuron.do_sample,
                temperature=self.config.neuron.temperature,
                top_k=self.config.neuron.top_k,
                top_p=self.config.neuron.top_p,
            ).query,
            kwargs=dict(message=prompt, role="user", disregard_system_prompt=False),
        )

        thread.start()

        bt.logging.debug(f"💬 Querying zephyr: {prompt}")
        token_streamer = partial(_forward, self.BATCH_SIZE, self.streamer)

        return synapse.create_streaming_response(token_streamer)
