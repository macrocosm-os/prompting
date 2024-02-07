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
# Bittensor Miner Template:
import prompting
from prompting.protocol import PromptingSynapse
# import base miner class which takes care of most of the boilerplate
from neurons.miner import Miner

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI



class OpenAIMiner(Miner):
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
            self.identity_tags =  ("openai_miner", ) + (self.config.neuron.model_id, )
        
        _ = load_dotenv(find_dotenv()) 
        api_key = os.environ.get("OPENAI_API_KEY")        
        self.client = OpenAI(api_key=api_key)
        # Set openai key and other args
        self.model = self.config.neuron.model_id

        bt.logging.info(f'Initialized {self.model}')

        self.system_prompt = "You are a friendly chatbot who always responds concisely and informatively."
        self.accumulated_total_tokens = 0
        self.accumulated_prompt_tokens = 0
        self.accumulated_completion_tokens = 0

    def get_cost_logging(self, usage):
        bt.logging.info(f"Total Tokens: {usage.total_tokens}")
        bt.logging.info(f"Prompt Tokens: {usage.prompt_tokens}")
        bt.logging.info(f"Completion Tokens: {usage.completion_tokens}")

        self.accumulated_total_tokens += usage.total_tokens
        self.accumulated_prompt_tokens += usage.prompt_tokens
        self.accumulated_completion_tokens += usage.completion_tokens

        return  {
            'total_tokens': usage.total_tokens,
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'accumulated_total_tokens': self.accumulated_total_tokens,
            'accumulated_prompt_tokens': self.accumulated_prompt_tokens,
            'accumulated_completion_tokens': self.accumulated_completion_tokens,
        }

    async def forward(
        self, synapse: PromptingSynapse
    ) -> PromptingSynapse:
        """
        Processes the incoming synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (PromptingSynapse): The synapse object containing the 'dummy_input' data.

        Returns:
            PromptingSynapse: The synapse object with the 'dummy_output' field set to twice the 'dummy_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        try:
            t0 = time.time()
            bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")
            role = synapse.roles[-1]
            message = synapse.messages[-1]
        
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": role, "content": message}
                ]
            )
            completion = response.choices[0].message.content

            synapse.completion = completion
            synapse_latency = time.time() - t0

            if self.config.wandb.on:
                    self.log_event(
                        timing=synapse_latency, 
                        prompt=message,
                        completion=completion,
                        system_prompt=self.system_prompt,
                        extra_info=self.get_cost_logging(response.usage)
                    )

            bt.logging.debug(f"✅ Served Response: {completion}")
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            synapse.completion = "Error: " + str(e)
        finally:
            if self.config.neuron.stop_on_forward_exception:
                self.should_exit = True
            return synapse


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with OpenAIMiner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)

            if miner.should_exit:
                bt.logging.warning("Ending miner...")
                break   
