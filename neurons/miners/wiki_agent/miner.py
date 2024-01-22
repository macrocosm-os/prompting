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

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv, find_dotenv
from langchain import OpenAI
from langchain.agents import Tool, initialize_agent
from agent import WikiAgent


class WikipediaAgentMiner(Miner):
    """Langchain-based miner which uses OpenAI's API as the LLM. This uses the ReAct framework.
    
    You should also install the dependencies for this miner, which can be found in the requirements.txt file in this directory.
    """
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

        parser.add_argument(
            "--wandb.on",
            type=bool,
            default=False,
            help="Enable wandb logging.",
        )

        parser.add_argument(
            "--wandb.entity",
            type=str,
            default="<<Add your wandb entity here>>",
            help="Wandb entity to log to.",
        )

        parser.add_argument(
            "--wandb.project_name",
            type=str,
            default="<<Add your wandb project name here>>",
            help="Wandb project to log to.",
        )


    def __init__(self, config=None):
        super().__init__(config=config)
        
        bt.logging.info(f"🤖📖 Initializing wikipedia agent with model {self.config.openai.model_name}...")

        if self.config.wandb.on:
            self.wandb_run.tags = self.wandb_run.tags + ("wikipedia_agent_miner", ) + (self.config.openai.model_name, )
        
        _ = load_dotenv(find_dotenv()) 
        
        
        self.agent = WikiAgent()


    async def forward(
        self, synapse: PromptingSynapse
    ) -> PromptingSynapse:
        """
        Processes the incoming synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (PromptingSynapse): The synapse object containing the 'dummy_input' data.

        Returns:
            PromptingSynapse: The synapse object with the '`dummy_output' field set to twice the 'dummy_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        try:
            t0 = time.time()
            bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")
                        
            message = synapse.messages[-1]
            
            bt.logging.debug(f"💬 Querying openai and wikipedia: {message}")
            
            response = self.agent.run(message)

            synapse.completion = response
            synapse_latency = time.time() - t0

            self.log_event(
                timing=synapse_latency, 
                prompt=message,
                completion=response,
                system_prompt=None
            )

            bt.logging.debug(f"✅ Served Response: {response}")
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with WikipediaAgentMiner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
