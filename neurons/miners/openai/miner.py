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

# Bittensor Miner Template:
import prompting
from prompting.protocol import PromptingSynapse
# import base miner class which takes care of most of the boilerplate
from neurons.miner import Miner

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI


class OpenAIMiner(Miner):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super().__init__(config=config)

        # Set openai key and other args
        self.model = ChatOpenAI(
            model_name="gpt-4",
            api_key=os.environ["OPENAI_API_KEY"],
            # **kwargs
        )

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

        prompt = ChatPromptTemplate.from_messages([("{role}", "{input}")])
        chain = prompt | self.model | StrOutputParser()

        response = chain.invoke(
            {"role": synapse.roles[-1], "input": synapse.messages[-1]}
        )

        synapse.response = response


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with OpenAIMiner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
