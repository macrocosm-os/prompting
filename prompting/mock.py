import time
import torch
import asyncio
import random
import bittensor as bt
from prompting.protocol import StreamPromptingSynapse

from functools import partial
from typing import List, Union, AsyncGenerator, Any, Iterator
from types import SimpleNamespace


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

        self.tokenizer = SimpleNamespace(tokenizer=MockTokenizer())
        self.phrase = phrase

    def __call__(self, messages):
        return self.forward(messages)

    def forward(self, messages):
        role_tag = self.tokenizer.tokenizer.role_expr.format(role="assistant")
        return f"{role_tag} {self.phrase}"


class MockPipeline:
    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def llm_engine(self):
        return SimpleNamespace(tokenizer=self.model.tokenizer)

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

    def __call__(self, composed_prompt, **kwargs):
        return self.forward(composed_prompt, **kwargs)

    def forward(self, messages, **kwargs):
        output = self.model(messages)
        return self.postprocess(output)

    def postprocess(self, output, **kwargs):
        output = output.split(self.model.tokenizer.tokenizer.role_expr.format(role="assistant"))[-1].strip()
        return output

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
    DEFAULT_IP = "127.0.0.0"
    DEFAULT_PORT = 8091

    def __init__(self, subtensor, netuid=1, network="mock"):
        super().__init__(netuid=netuid, network=network, sync=False)

        self.subtensor = subtensor
        self.sync(subtensor=self.subtensor)

        for axon in self.axons:
            axon.ip = self.DEFAULT_IP
            axon.port = self.DEFAULT_PORT


class MockStreamMiner:
    """MockStreamMiner is an echo miner"""

    MIN_DELAY_PERCENTAGE = 0.20  # 20%
    MAX_DELAY_PERCENTAGE = 0.50  # 50%

    def __init__(self, streaming_batch_size: int, timeout: float):
        self.streaming_batch_size = streaming_batch_size
        self.timeout = timeout

    def forward(self, synapse: StreamPromptingSynapse, start_time: float) -> Iterator:
        """Mock forward returns a token_streamer, which is a partial function
        that simulates the async streaming of tokens from the axon.

        In production, we actually return the synapse.create_streaming_response(token_streamer).
         > create_streaming_response enables communication between miner/validator via aiohttp post requests
         via a BTStreamingResponse.

        Returns:
            StreamPromptingSynapse: _description_
        """

        def _forward(self, prompt: str, start_time: float, sample: Any):
            """In production, _forward is an async def _forward. This is because we are sending an
            aiohttp post request to the axon to get chunks of data. This is the "send" packet defined
            in typical _forward functions.

            Here, we simulate streaming by iterating through a prompt and stochastically delaying.
            """
            buffer = []
            continue_streaming = True

            try:
                for token in prompt.split():  # split on spaces.
                    buffer.append(token)

                    if time.time() - start_time > self.timeout:
                        print(f"⏰ Timeout reached, stopping streaming. {time.time() - self.start_time}")
                        break

                    if len(buffer) == self.streaming_batch_size:
                        time.sleep(
                            self.timeout * random.uniform(self.MIN_DELAY_PERCENTAGE, self.MAX_DELAY_PERCENTAGE)
                        )  # simulate some async processing time
                        yield buffer, continue_streaming
                        buffer = []

                if buffer:
                    continue_streaming = False
                    yield buffer, continue_streaming

            except Exception as e:
                bt.logging.error(f"Error in forward: {e}")

        prompt = synapse.messages[-1]
        token_streamer = partial(_forward, self, prompt, start_time)

        return token_streamer


class MockDendrite(bt.dendrite):
    """
    Replaces a real bittensor network request with a mock request that just returns some static
    completion for all axons that are passed and adds some random delay.
    """

    MIN_TIME: float = 0
    MAX_TIME: float = 1

    def __init__(self, wallet):
        super().__init__(wallet)

    async def call(
        self,
        i: int,
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12.0,
        deserialize: bool = True,
    ) -> bt.Synapse:
        """Simulated call method to fill synapses with mock data."""

        process_time = random.random() * (self.MAX_TIME - self.MIN_TIME) + self.MIN_TIME

        if process_time < timeout:
            # Update the status code and status message of the dendrite to match the axon
            synapse.completion = f"Mock miner completion {i}"
            synapse.dendrite.status_code = 200
            synapse.dendrite.status_message = "OK"
        else:
            synapse.completion = ""
            synapse.dendrite.status_code = 408
            synapse.dendrite.status_message = "Timeout"

        synapse.dendrite.process_time = str(process_time)

        # Return the updated synapse object after deserializing if requested
        if deserialize:
            return synapse.deserialize()
        else:
            return synapse

    async def call_stream(
        self,
        synapse: StreamPromptingSynapse,
        timeout: float = 12.0,
        deserialize: bool = True,
    ) -> AsyncGenerator[Any, Any]:
        """
        Yields:
            object: Each yielded object contains a chunk of the arbitrary response data from the Axon.
            bittensor.Synapse: After the AsyncGenerator has been exhausted, yields the final filled Synapse.

            Communications delay is simulated in the MockStreamMiner.forward method. Therefore, we can
            compute the process_time directly here.
        """

        start_time = time.time()
        continue_streaming = True
        response_buffer = []

        miner = MockStreamMiner(streaming_batch_size=12, timeout=timeout)
        token_streamer = miner.forward(synapse, start_time)

        # Simulating the async streaming without using aiohttp post request
        while continue_streaming:
            for buffer, continue_streaming in token_streamer(True):
                response_buffer.extend(buffer)  # buffer is a List[str]
                process_time = time.time() - start_time

                if not continue_streaming:
                    synapse.dendrite.process_time = process_time
                    synapse.completion = " ".join(response_buffer)
                    synapse.dendrite.status_code = 200
                    synapse.dendrite.status_message = "OK"
                    response_buffer = []
                    break

                elif process_time >= timeout:
                    synapse.completion = " ".join(response_buffer)  # partially completed response buffer
                    synapse.dendrite.status_code = 408
                    synapse.dendrite.status_message = "Timeout"
                    synapse.dendrite.process_time = timeout
                    continue_streaming = False  # to stop the while loop
                    break

        # Return the updated synapse object after deserializing if requested
        if deserialize:
            yield synapse.deserialize()
        else:
            yield synapse

    async def forward(
        self,
        axons: List[bt.axon],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
    ):
        assert isinstance(
            synapse, StreamPromptingSynapse
        ), "Synapse must be a StreamPromptingSynapse object when is_stream is True."

        async def query_all_axons(is_stream: bool):
            """Queries all axons for responses."""

            async def single_axon_response(i: int, target_axon: Union[bt.AxonInfo, bt.axon]):
                """Queries a single axon for a response."""

                s = synapse.copy()

                target_axon = target_axon.info() if isinstance(target_axon, bt.axon) else target_axon

                # Attach some more required data so it looks real
                s = self.preprocess_synapse_for_request(target_axon_info=target_axon, synapse=s, timeout=timeout)

                if is_stream:
                    # If in streaming mode, return the async_generator
                    return self.call_stream(
                        synapse=s,
                        timeout=timeout,
                        deserialize=False,
                    )
                else:
                    return await self.call(
                        i=i,
                        synapse=s,
                        timeout=timeout,
                        deserialize=deserialize,
                    )

            if not run_async:
                return [await single_axon_response(target_axon) for target_axon in axons]

            # If run_async flag is True, get responses concurrently using asyncio.gather().
            return await asyncio.gather(*(single_axon_response(i, target_axon) for i, target_axon in enumerate(axons)))

        return await query_all_axons(is_stream=streaming)

    def __str__(self) -> str:
        """
        Returns a string representation of the Dendrite object.

        Returns:
            str: The string representation of the Dendrite object in the format "dendrite(<user_wallet_address>)".
        """
        return "MockDendrite({})".format(self.keypair.ss58_address)
