import time
import uuid
import torch
import asyncio
import random
import bittensor as bt

from typing import List


class MockTokenizer:
    def __init__(self):
        super().__init__()

        self.role_expr = "<|mock-{role}|>"

    def apply_chat_template(self, messages, **kwargs):
        prompt = ""
        for m in messages:
            role = self.role_expr.format(role=m["role"])
            content = m["content"]
            prompt += f"<|mock-{role}|> {content}\n"

        return "\n".join(prompt)


class MockModel(torch.nn.Module):
    def __init__(self, phrase):
        super().__init__()

        self.tokenizer = MockTokenizer()
        self.phrase = phrase

    def __call__(self, messages):
        return self.forward(messages)

    def forward(self, messages):
        role_tag = self.tokenizer.role_expr.format(role="assistant")
        return f"{role_tag} {self.phrase}"


class MockPipeline:
    @property
    def tokenizer(self):
        return self.model.tokenizer

    def __init__(
        self,
        phrase="Mock llm output",
        model_kwargs=None,
    ):
        super().__init__()

        self.model_kwargs = model_kwargs or {}
        self.model = MockModel(phrase)

    def __repr__(self):
        return f"{self.__class__.__name__}(phrase={self.model.phrase})"

    def __call__(self, messages, **kwargs):
        return self.forward(messages, **kwargs)

    def forward(self, messages, **kwargs):
        output = self.model(messages)
        return self.postprocess(output)

    def postprocess(self, output, **kwargs):
        output = output.split(self.model.tokenizer.role_expr.format(role="assistant"))[
            -1
        ].strip()
        return [{"generated_text": output}]

    def preprocess(self, **kwargs):
        pass


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None):

        super().__init__()
        # reset the underlying subtensor state
        self.chain_state = None
        self.setup()

        if not self.subnet_exists(netuid):
            self.create_subnet(netuid)

        # Register ourself (the validator) as a neuron at uid=0
        if wallet is not None:
            self.force_register_neuron(
                netuid=netuid,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                balance=100000,
                stake=100000,
            )

        # Register n mock neurons who will be miners
        for i in range(1, n + 1):
            self.force_register_neuron(
                netuid=netuid,
                hotkey=f"miner-hotkey-{i}",
                coldkey="mock-coldkey",
                balance=100000,
                stake=100000,
            )


class MockMetagraph(bt.metagraph):

    default_ip = "127.0.0.0"
    default_port = 8091

    def __init__(self, netuid=1, network="mock", subtensor=None):
        super().__init__(netuid=netuid, network=network, sync=False)

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = self.default_ip
            axon.port = self.default_port


class MockDendrite(bt.dendrite):
    """
    Replaces a real bittensor network request with a mock request that just returns some static completion for all axons that are passed and adds some random delay.
    """

    min_time: float = 0
    max_time: float = 1

    def __init__(self, wallet):
        super().__init__(wallet)

    async def forward(
        self,
        axons: List[bt.axon],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ):

        if streaming:
            raise NotImplementedError("Streaming not implemented yet.")

        async def query_all_axons(streaming: bool):
            """Queries all axons for responses."""

            async def single_axon_response(i, axon):
                """Queries a single axon for a response."""

                t0 = time.time()
                s = synapse.copy()
                # Attach some more required data so it looks real
                s = self.preprocess_synapse_for_request(axon, s, timeout)
                # We just want to mock the response, so we'll just fill in some data
                process_time = (
                    random.random() * (self.max_time - self.min_time) + self.min_time
                )
                await asyncio.sleep(process_time)
                if process_time < timeout:
                    # Update the status code and status message of the dendrite to match the axon
                    s.completion = f"Mock miner completion {i}"
                    s.dendrite.status_code = 200
                    s.dendrite.status_message = "OK"
                else:
                    s.completion = ""
                    s.dendrite.status_code = 408
                    s.dendrite.status_message = "Timeout"

                s.dendrite.process_time = str(time.time() - t0)

                # Return the updated synapse object after deserializing if requested
                if deserialize:
                    return s.deserialize()
                else:
                    return s

            return await asyncio.gather(
                *(
                    single_axon_response(i, target_axon)
                    for i, target_axon in enumerate(axons)
                )
            )

        return await query_all_axons(streaming)

    def __str__(self) -> str:
        """
        Returns a string representation of the Dendrite object.

        Returns:
            str: The string representation of the Dendrite object in the format "dendrite(<user_wallet_address>)".
        """
        return "MockDendrite({})".format(self.keypair.ss58_address)
