# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback


def get_cost_logging(cb) -> dict:        
    bt.logging.info(f"Total Tokens: {cb.total_tokens}")
    bt.logging.info(f"Prompt Tokens: {cb.prompt_tokens}")
    bt.logging.info(f"Completion Tokens: {cb.completion_tokens}")
    bt.logging.info(f"Total Cost (USD): ${cb.total_cost}")

    return  {
        'total_tokens': cb.total_tokens,
        'prompt_tokens': cb.prompt_tokens,
        'completion_tokens': cb.completion_tokens,
        'total_cost': cb.total_cost,
    }
    


class OpenAIMiner(Miner):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds OpenAI-specific arguments to the command line parser.
        """
        super().add_args(parser)
        parser.add_argument(
            "--openai.model_name",
            type=str,
            default="gpt-4",
            help="OpenAI model to use for completion.",
        )

    

    def __init__(self, config=None):
        super().__init__(config=config)
        
        bt.logging.info(f"Initializing with model {self.config.openai.model_name}...")

        if self.config.wandb.on:
            self.wandb_run.tags = self.wandb_run.tags + ("openai_miner", ) + (self.config.openai.model_name, )
        
        _ = load_dotenv(find_dotenv()) 
        api_key = os.environ.get("OPENAI_API_KEY")        

        # Set openai key and other args
        self.model = ChatOpenAI(
            model_name=self.config.openai.model_name,
            api_key=api_key
        )

        self.system_prompt = "You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know."


 

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
        # TODO(developer): Replace with actual implementation logic.
        try:

            t0 = time.time()
            bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")

            with get_openai_callback() as cb:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.system_prompt),
                    ("user", "{input}")
                ])
                chain = prompt | self.model | StrOutputParser()

                role = synapse.roles[-1]
                message = synapse.messages[-1]
                
                bt.logging.debug(f"💬 Querying openai: {prompt}")
                response = chain.invoke(
                    {"role": role, "input": message}
                )

                synapse.completion = response
                synapse_latency = time.time() - t0                               

                if self.config.wandb.on:
                    self.log_event(
                        timing=synapse_latency, 
                        prompt=message,
                        completion=response,
                        system_prompt=self.system_prompt,
                        extra_info=get_cost_logging(cb)
                    )


            bt.logging.debug(f"✅ Served Response: {response}")
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with OpenAIMiner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
