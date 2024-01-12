import torch
import bittensor as bt

from transformers import Pipeline


class MockTokenizer(torch.nn.Module):
    def __init__(self):
        super(MockTokenizer, self).__init__()

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
        super(MockModel, self).__init__()

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
        model_id="mock",
        device_map="mock-cuda",
        torch_dtype="torch.mock16",
        phrase="mock reply",
        model_kwargs=None,
    ):
        super(MockPipeline, self).__init__()

        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.model_kwargs = model_kwargs or {}
        self.model = MockModel(phrase)

    def __repr__(self):
        return f"{self.__class__.__name__}(model_id={self.model_id}, device_map={self.device_map}, torch_dtype={self.torch_dtype})"

    def __call__(self, messages, **kwargs):
        return self.forward(messages, **kwargs)

    def forward(self, messages, **kwargs):
        output = self.model(messages)
        return self.postprocess(output)

    def postprocess(self, output, **kwargs):
        output = output.split(
            self.model.tokenizer.role_expr.format(role="assistant")
        )[-1].strip()
        return [{"generated_text": output}]

    def preprocess(self, **kwargs):
        pass


class MockSubtensor(bt.MockSubtensor):
    def __init__(self, netuid, n=16, wallet=None, network="mock"):
        super(MockSubtensor, self).__init__(network=network)

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
                coldkey="mock123",
                balance=100000,
                stake=100000,
            )


class MockMetagraph(bt.metagraph):
    def __init__(self, netuid=1, network="mock", subtensor=None):
        super(MockMetagraph, self).__init__(
            netuid=netuid, network=network, sync=False
        )

        if subtensor is not None:
            self.subtensor = subtensor
        self.sync(subtensor=subtensor)

        for axon in self.axons:
            axon.ip = "1.2.3.4"
            axon.port = 8008

        bt.logging.info(f"Metagraph: {self}")
        bt.logging.info(f"Axons: {self.axons}")
