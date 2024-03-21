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
import threading
from starlette.types import Send
from typing import Awaitable

# Bittensor Miner Template:
from prompting.protocol import StreamPromptingSynapse
from prompting.llm import load_pipeline
from prompting.llm import HuggingFaceLLM

# import base miner class which takes care of most of the boilerplate
from prompting.base.prompting_miner import BaseStreamPromptingMiner
from prompting.llm import CustomTextIteratorStreamer
import asyncio


class HuggingFaceMiner(BaseStreamPromptingMiner):
    """
    Base miner which runs zephyr (https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
    This requires a GPU with at least 20GB of memory.
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
            torch_dtype=torch.bfloat16,
            device=self.device,
            mock=mock,
            return_streamer=True,
            model_kwargs=model_kwargs,
        )

        self.model_id = self.config.neuron.model_id
        self.system_prompt = self.config.neuron.system_prompt
        self.lock = asyncio.Lock()
        self.loop = asyncio.new_event_loop()
        # Running the loop in a separate thread allows it to be used by
        # synchronous functions without blocking the main thread.
        self.thread = threading.Thread(target=self.start_loop, args=(self.loop,))
        self.thread.start()

    def start_loop(self, loop):
        """Run the event loop in a separate thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def stop(self):
        """Stop the event loop and wait for the thread to finish."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
    
    async def start_llm_generation(self, prompt, timeout = 5):
        bt.logging.debug(f"🛑 Attempting to acquire the lock for llm in {timeout} seconds - task_id: {id(asyncio.current_task())}")
        
        try:                        
            await asyncio.wait_for(self.lock.acquire(), timeout)            
            bt.logging.debug("🔒 Lock acquired. Accessing the shared resource...")                        
            bt.logging.debug('📦 Starting llm generation, populating streamer...')
            response = HuggingFaceLLM(
                    llm_pipeline=self.llm_pipeline,
                    system_prompt=self.system_prompt,
                    max_new_tokens=self.config.neuron.max_tokens,
                    do_sample=self.config.neuron.do_sample,
                    temperature=self.config.neuron.temperature,
                    top_k=self.config.neuron.top_k,
                    top_p=self.config.neuron.top_p,
                ).query(message=prompt, role="user", disregard_system_prompt=False)
            
            bt.logging.debug('🧼 Generation completed, cleaning streamer...')
            if self.streamer.has_data():
                self.streamer.clear_queue()
            
            bt.logging.debug("Cleaning cuda cache")
            torch.cuda.empty_cache()             
            return response     
                   
        except asyncio.TimeoutError:
            bt.logging.error(f"Could not access the shared resource within {timeout} seconds.")
            
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            
        finally:
            bt.logging.debug(f"🔓 Releasing lock from task_id {id(asyncio.current_task())}") 
            self.lock.release()
            bt.logging.success(f"🟢 Semaphore released from task_id {id(asyncio.current_task())}")
                       
            
                    
    def forward(self, synapse: StreamPromptingSynapse) -> Awaitable:
        async def _forward(
            self,
            prompt: str,            
            init_time: float,
            timeout_threshold: float,
            task,
            loop,
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

            buffer = []
            temp_completion = ""  # for wandb logging
            timeout_reached = False
            system_message = ""
            
            # loop.run_until_complete(task)
            asyncio.run_coroutine_threadsafe(task, self.loop)
            bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")
                                    
            try:
                synapse_message = synapse.messages[-1]    
                for token in streamer:
                    system_message += token                                        
                    
                    buffer.append(token)                    
                    system_message += "".join(buffer)
                    
                    if synapse_message in system_message:
                        # Cleans system message and challenge from model response
                        bt.logging.debug(f"Discarding initial system_prompt / user prompt inputs from generation...")
                        buffer=[]
                        system_message = ""
                        continue
                    

                    if time.time() - init_time > timeout_threshold:
                        bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
                        timeout_reached = True
                        break

                    if len(buffer) == self.config.neuron.streaming_batch_size:
                        joined_buffer = "".join(buffer)
                        temp_completion += joined_buffer                        

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
                #_ = task.result() # wait for thread to finish          
                bt.logging.debug('Finishing streaming loop...')
                bt.logging.debug('-' * 50)
                bt.logging.debug(f'---->>> Received message:')
                bt.logging.debug(synapse.messages[0])
                bt.logging.debug('-' * 50)
                bt.logging.debug(f'<<<----- Returned message:')
                bt.logging.debug(temp_completion)
                bt.logging.debug('-' * 50)                  
                synapse_latency = time.time() - init_time
                
                if self.config.wandb.on:
                    self.log_event(
                        timing=synapse_latency,
                        prompt=prompt,
                        completion=temp_completion,
                        system_prompt=self.system_prompt,
                    )

        # bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")
        prompt = synapse.messages[-1]


        task = self.start_llm_generation(prompt)

        init_time = time.time()
        timeout_threshold = synapse.timeout

        token_streamer = partial(
            _forward,
            self,
            prompt,            
            init_time,
            timeout_threshold,
            task,
            self.loop,
            self.streamer            
        )        
                
        return synapse.create_streaming_response(token_streamer)
