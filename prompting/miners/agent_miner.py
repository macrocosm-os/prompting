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
import bittensor as bt
import argparse
from deprecation import deprecated

# Bittensor Miner Template:
from prompting.protocol import PromptingSynapse

# import base miner class which takes care of most of the boilerplate
from prompting.base.prompting_miner import BasePromptingMiner
from dotenv import load_dotenv, find_dotenv
from prompting.miners.agents import SingleActionAgent, ReactAgent
from langchain.callbacks import get_openai_callback

@deprecated(deprecated_in="1.1.2", removed_in="2.0", details="AgentMiner is unsupported.")
class AgentMiner(BasePromptingMiner):
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
            "--use_react_agent",
            type=bool,
            default=False,
            help="Flag to enable the ReAct agent",
        )

    def __init__(self, config=None):
        super().__init__(config=config)

        bt.logging.info(
            f"🤖📖 Initializing wikipedia agent with model {self.config.neuron.model_id}..."
        )

        if self.config.wandb.on:
            self.identity_tags = ("wikipedia_agent_miner",) + (
                self.config.neuron.model_id,
            )

        _ = load_dotenv(find_dotenv())

        if self.config.use_react_agent:
            self.agent = ReactAgent(
                self.config.neuron.model_id,
                self.config.neuron.temperature,
                self.config.neuron.max_tokens,
                self.config.neuron.load_in_8bits,
                self.config.neuron.load_in_4bits,
            )
        else:
            self.agent = SingleActionAgent(
                self.config.neuron.model_id,
                self.config.neuron.temperature,
                self.config.neuron.max_tokens,
                self.config.neuron.load_in_8bits,
                self.config.neuron.load_in_4bits,
            )

        self.accumulated_total_tokens = 0
        self.accumulated_prompt_tokens = 0
        self.accumulated_completion_tokens = 0
        self.accumulated_total_cost = 0

    def get_cost_logging(self, cb):
        bt.logging.info(f"Total Tokens: {cb.total_tokens}")
        bt.logging.info(f"Prompt Tokens: {cb.prompt_tokens}")
        bt.logging.info(f"Completion Tokens: {cb.completion_tokens}")
        bt.logging.info(f"Total Cost (USD): ${cb.total_cost}")

        self.accumulated_total_tokens += cb.total_tokens
        self.accumulated_prompt_tokens += cb.prompt_tokens
        self.accumulated_completion_tokens += cb.completion_tokens
        self.accumulated_total_cost += cb.total_cost

        return {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost": cb.total_cost,
            "accumulated_total_tokens": self.accumulated_total_tokens,
            "accumulated_prompt_tokens": self.accumulated_prompt_tokens,
            "accumulated_completion_tokens": self.accumulated_completion_tokens,
            "accumulated_total_cost": self.accumulated_total_cost,
        }

    async def forward(self, synapse: PromptingSynapse) -> PromptingSynapse:
        self.should_exit = True 
        return synapse 
