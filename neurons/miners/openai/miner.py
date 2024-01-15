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

import time
import typing
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


class OpenAIMiner(Miner):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds OpenAI-specific arguments to the command line parser.
        """
        parser.add_argument(
            "--openai.model_name",
            type=str,
            default="gpt-4",
            help="OpenAI model to use for completion.",
        )

        parser.add_argument(
            "--wandb.on",
            type=bool,
            default=True,
            help="Enable wandb logging.",            
        )

        parser.add_argument(
            "--wandb.entity",
            type=str,
            default="sn1",
            help="wandb entity to log to.",
        )

        parser.add_argument(
            "--wandb.project",
            type=str,
            default="miners",
        )

    
    def config(self) -> bt.config:
        """
        Provides the configuration for the OpenAIMiner.

        This method returns a configuration object specific to the OpenAIMiner, containing settings
        and parameters related to the OpenAI model and its interaction parameters. The configuration
        ensures the miner's optimal operation with the OpenAI model and can be customized by adjusting
        the command-line arguments introduced in the `add_args` method.

        Returns:
            bittensor.Config:
                A configuration object specific to the OpenAIMiner, detailing the OpenAI model settings
                and operational parameters.

        Note:
            If introducing new settings or parameters for OpenAI or the miner's operation, ensure they
            are properly initialized and returned in this configuration method.
        """
        parser = argparse.ArgumentParser(description="OpenAI Miner Configs")
        self.add_args(parser)
        return bt.config(parser)

    

    def __init__(self, config=None):
        super().__init__(config=config)
        
        bt.logging.info(f"Initializing with model {config.openai.model_name}...")

        if self.config.wandb.on:
            self.wandb_run.tags = self.wandb_run.tags + ["openai_miner", config.openai.model_name]

        # Set openai key and other args
        self.model = ChatOpenAI(
            model_name=self.config.openai.model_name,
            api_key="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",
            # **kwargs
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
            bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")

            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("user", "{input}")
            ])
            chain = prompt | self.model | StrOutputParser()

            bt.logging.debug(f"💬 Querying openai: {prompt}")
            response = chain.invoke(
                {"role": synapse.roles[-1], "input": synapse.messages[-1]}
            )

            synapse.completion = response

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
