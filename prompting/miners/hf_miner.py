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
import torch
import argparse
import bittensor as bt
from functools import partial
from threading import Thread
from starlette.types import Send
from typing import Dict, List

# Bittensor Miner Template:
from prompting.protocol import StreamPromptingSynapse
from prompting.llm import load_pipeline
from prompting.llm import HuggingFaceLLM

# import base miner class which takes care of most of the boilerplate
from prompting.base.prompting_miner import BaseStreamPromptingMiner
from prompting.llm import CustomTextIteratorStreamer


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

        # Forces model loading behaviour over mock flag
        mock = (
            False if self.config.neuron.should_force_model_loading else self.config.mock
        )

        self.llm_pipeline, self.streamer = load_pipeline(
            model_id=self.config.neuron.model_id,
            device=self.device,
            mock=mock,
            return_streamer=True,
            model_kwargs=model_kwargs,
        )

        self.model_id = self.config.neuron.model_id
        self.system_prompt = self.config.neuron.system_prompt

    def forward(self, synapse: StreamPromptingSynapse) -> StreamPromptingSynapse:
        async def _forward(
            self,
            prompt: str,
            thread: Thread,
            init_time: float,
            timeout_threshold: float,
            streamer: CustomTextIteratorStreamer,
            send: Send,
        ):
            """_summary_

            Args:
                prompt (str): The received message (challenge) in the synapse. For logging.
                thread (Thread): A background thread that is reponsible for running the model.
                init_time (float): Initial time of the forward call. For timeout calculation.
                timeout_threshold (float): The amount of time that the forward call is allowed to run. If timeout is reached, streaming stops and
                    validators recieve a partial response.
                streamer (CustomTextIteratorStreamer): Iterator that holds tokens within a background Queue to be returned when sampled.
                send (Send): bittensor aiohttp send function to send the response back to the validator.
            """
            bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")

            buffer = []
            temp_completion = ""  # for wandb logging
            timeout_reached = False

            try:
                for token in streamer:
                    buffer.append(token)

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
                    temp_completion += joined_buffer
                    bt.logging.debug(f"Streamed tokens: {joined_buffer}")

                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": False,
                        }
                    )

            except Exception as e:
                bt.logging.error(f"Error in forward: {e}")
                if self.config.neuron.stop_on_forward_exception:
                    self.should_exit = True

            finally:
                # Thread and streamer cleanup
                thread.join()  # This will close the thread but not stop execution of the model
                if streamer.has_data():
                    streamer.clear_queue()  # model continues to compute, so remove tokens in queue

                synapse_latency = time.time() - init_time

                if self.config.wandb.on:
                    self.log_event(
                        timing=synapse_latency,
                        prompt=prompt,
                        completion=temp_completion,
                        system_prompt=self.system_prompt,
                    )

                torch.cuda.empty_cache()  # cuda cleanup

        bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")
        prompt = synapse.messages[-1]

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

        bt.logging.debug(f"💬 Querying hf-miner: {prompt}")

        init_time = time.time()
        timeout_threshold = synapse.timeout

        token_streamer = partial(
            _forward,
            self,
            prompt,
            thread,
            init_time,
            timeout_threshold,
            self.streamer,
        )

        return synapse.create_streaming_response(token_streamer)
