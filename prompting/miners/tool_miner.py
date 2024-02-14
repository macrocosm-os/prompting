import os
import typing
import argparse
import bittensor as bt
import wikipedia
import time

# Bittensor Miner Template:
from prompting.protocol import PromptingSynapse

# import base miner class which takes care of most of the boilerplate
from prompting.base.prompting_miner import BasePromptingMiner
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from traceback import print_exception


class ToolMiner(BasePromptingMiner):
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        bt.logging.info(f"Initializing with model {self.config.neuron.model_id}...")

        if self.config.wandb.on:
            self.identity_tags = ("openai_miner",) + (self.config.neuron.model_id,)

        _ = load_dotenv(find_dotenv())
        api_key = os.environ.get("OPENAI_API_KEY")

        # Set openai key and other args
        self.model = ChatOpenAI(
            api_key=api_key,
            model_name=self.config.neuron.model_id,
            max_tokens=self.config.neuron.max_tokens,
            temperature=self.config.neuron.temperature,
        )

        self.system_prompt = """You are a nice AI assistant that uses the provided context to answer user queries.
        ## Context
        {context}
        """

    async def forward(self, synapse: PromptingSynapse) -> PromptingSynapse:
        try:
            with get_openai_callback() as cb:
                t0 = time.time()
                bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")

                role = synapse.roles[-1]
                message = synapse.messages[-1]

                # Message needs to be limited to 300 characters for wikipedia search, otherwise it will a return an error
                matches = wikipedia.search(message[:300])

                # If we find a match, we add the context to the system prompt
                if len(matches) > 0:
                    title = matches[0]
                    page = wikipedia.page(title)
                    context = page.content

                    if len(context) > 12_000:
                        context = context[:12_000]

                    formatted_system_prompt = self.system_prompt.format(context=context)
                else:
                    formatted_system_prompt = self.config.neuron.system_prompt

                prompt = ChatPromptTemplate.from_messages(
                    [("system", formatted_system_prompt), ("user", "{input}")]
                )
                chain = prompt | self.model | StrOutputParser()

                bt.logging.debug(f"💬 Querying openai: {prompt}")
                response = chain.invoke({"role": role, "input": message})

                synapse.completion = response
                synapse_latency = time.time() - t0

                if self.config.wandb.on:
                    self.log_event(
                        timing=synapse_latency,
                        prompt=message,
                        completion=response,
                        system_prompt=self.system_prompt,
                        extra_info=self.get_cost_logging(cb),
                    )

            bt.logging.debug(f"✅ Served Response: {response}")
            return synapse
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            bt.logging.error(print_exception(value=e))
            synapse.completion = "Error: " + str(e)
        finally:
            if self.config.neuron.stop_on_forward_exception:
                self.should_exit = True
            return synapse

    async def blacklist(self, synapse: PromptingSynapse) -> typing.Tuple[bool, str]:
        return False, "All good here"

    async def priority(self, synapse: PromptingSynapse) -> float:
        return 1e6
