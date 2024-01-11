import time

import bittensor as bt

from transformers import Pipeline, pipeline
from prompting.mock import MockPipeline
from abc import ABC, abstractmethod


def pipeline(model_id, device_map=None, torch_dtype=None, mock=False):
    if mock:
        return MockPipeline(model_id, device_map=device_map, torch_dtype=torch_dtype)

    return pipeline(
        "text-generation",
        model=model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

class LLM(ABC):
    def __init__(self, pipeline, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95):
        self.pipeline = pipeline
        self.kwargs = dict(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        self.messages = []
        self.times = []

    @abstractmethod
    def query(self, message, cleanup=True, role="user"):
        pass

    @abstractmethod
    def forward(self, messages, cleanup=False):
        pass

    def _make_prompt(self, messages):
        # Abstract method for creating the prompt from messages
        pass

    def __call__(self, messages):
        return self.forward(messages)





class HuggingFaceLLM(LLM):

    def __init__(
        self,
        pipeline: Pipeline,
        system_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    ):
        self.pipeline = pipeline
        self.system_prompt = system_prompt
        self.kwargs = dict(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        self.messages = [{"content": self.system_prompt, "role": "system"}]
        self.times = [0]

    def query(self, message, cleanup=True, role="user", disregard_system_prompt=False):
        messages = self.messages + [{"content": message, "role": role}]

        if disregard_system_prompt:
            messages = messages[1:]

        tbeg = time.time()
        response = self.forward(messages, cleanup=cleanup)

        self.messages = messages + [{"content": response, "role": "assistant"}]
        self.times = self.times + [0, time.time() - tbeg]
        return response

    def __call__(self, messages):
        return self.forward(messages)

    def _make_prompt(self, messages):
        return self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def forward(self, messages, cleanup=False, preformat_messages=False):
        prompt = self._make_prompt(messages)
        outputs = self.pipeline(prompt, **self.kwargs)
        bt.logging.info(f"Generated response: {outputs}")
        response = outputs[0]["generated_text"]

        response = response.replace(prompt, "").strip()
        response.split("\n")
        if cleanup and response.startswith("Assistant:"):
            print(f"Cleaning up response: {response}")
            response = response.strip("Assistant:").split("User:")[0].strip("\n")

        return response

