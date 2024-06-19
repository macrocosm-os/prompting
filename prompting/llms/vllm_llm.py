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
import gc
import time
import torch
import bittensor as bt
from typing import List, Dict, Optional, Any
from vllm import LLM, SamplingParams
from prompting.cleaners.cleaner import CleanerPipeline
from prompting.llms import BasePipeline, BaseLLM
from prompting.mock import MockPipeline
from prompting.llms.utils import calculate_gpu_requirements


def load_vllm_pipeline(model_id: str, device: str, gpus: int, max_allowed_memory_in_gb: int, mock=False):
    """Loads the VLLM pipeline for the LLM, or a mock pipeline if mock=True"""
    if mock or model_id == "mock":
        return MockPipeline(model_id)

    # Calculates the gpu memory utilization required to run the model.
    max_allowed_memory_allocation_in_bytes = max_allowed_memory_in_gb * 1e9
    gpu_mem_utilization = calculate_gpu_requirements(
        device, gpus, max_allowed_memory_allocation_in_bytes
    )

    try:
        # Attempt to initialize the LLM
        llm = LLM(model=model_id, gpu_memory_utilization = gpu_mem_utilization, quantization="AWQ", tensor_parallel_size=gpus)
        # This solution implemented by @bkb2135 sets the eos_token_id directly for efficiency in vLLM usage.
        # This approach avoids the overhead of loading a tokenizer each time the custom eos token is needed.
        # Using the Hugging Face pipeline, the eos token specific to llama models was fetched and saved (128009).
        # This method provides a straightforward solution, though there may be more optimal ways to manage custom tokens.
        llm.llm_engine.tokenizer.eos_token_id = 128009
        return llm
    except Exception as e:
        bt.logging.error(
            f"Error loading the VLLM pipeline within {max_allowed_memory_in_gb}GB: {e}"
        )
        raise e
        


class vLLMPipeline(BasePipeline):
    def __init__(
        self,
        model_id: str,
        llm_max_allowed_memory_in_gb: int,
        device: str = None,
        gpus: int = 1,
        mock: bool = False
    ):
        super().__init__()
        self.llm = load_vllm_pipeline(model_id, device, gpus, llm_max_allowed_memory_in_gb, mock)
        self.mock = mock
        self.gpus = gpus
        self.tokenizer = self.llm.llm_engine.tokenizer.tokenizer

    def __call__(self, composed_prompt: str, **model_kwargs: Dict) -> str:
        if self.mock:
            return self.llm(composed_prompt, **model_kwargs)

        # Compose sampling params
        temperature = model_kwargs.get("temperature", 0.8)
        top_p = model_kwargs.get("top_p", 0.95)
        max_tokens = model_kwargs.get("max_tokens", 256)

        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        output = self.llm.generate(composed_prompt, sampling_params, use_tqdm=True)
        response = output[0].outputs[0].text
        return response


class vLLM_LLM(BaseLLM):
    def __init__(
        self,
        llm_pipeline: BasePipeline,
        system_prompt,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
    ):
        model_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        super().__init__(llm_pipeline, system_prompt, model_kwargs)

        # Keep track of generation data using messages and times
        self.messages = [{"content": self.system_prompt, "role": "system"}]
        self.times: List[float] = [0]
        self._role_template = {
            "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "end": "<|start_header_id|>assistant<|end_header_id|>",
        }

    def query_conversation(
        self,
        messages: list[str],
        roles: list[str],
        cleaner: Optional[CleanerPipeline] = None,
    ):
        """Query LLM with the given lists of conversation history and roles."""
        assert len(messages) == len(roles), "Length of messages and roles must be the same"
        inputs: list[dict[str, Any]] = [{"content": self.system_prompt, "role": "system"}]
        for role, message in zip(roles, messages):
            inputs.append({"content": message, "role": role})

        t0 = time.perf_counter()
        response = self.forward(messages=inputs)
        response = self.clean_response(cleaner, response)
        self.times.extend((0, time.perf_counter() - t0))
        return response

    def query(
        self,
        message: str,
        role: str = "user",
        disregard_system_prompt: bool = False,
        cleaner: CleanerPipeline = None,
    ):
        # Adds the message to the list of messages for tracking purposes, even though it's not used downstream
        messages = self.messages + [{"content": message, "role": role}]

        t0 = time.time()
        response = self.forward(messages=messages)
        response = self.clean_response(cleaner, response)

        self.messages = messages
        self.messages.append({"content": response, "role": "assistant"})
        self.times.extend((0, time.time() - t0))

        return response

    def _make_prompt(self, messages: List[Dict[str, str]]) -> str:
        composed_prompt: List[str] = []

        for message in messages:
            role = message["role"]
            if role not in self._role_template:
                continue
            content = message["content"]
            composed_prompt.append(self._role_template[role].format(content))

        # Adds final tag indicating the assistant's turn
        composed_prompt.append(self._role_template["end"])
        return "".join(composed_prompt)

    def forward(self, messages: List[Dict[str, str]]):
        # make composed prompt from messages
        composed_prompt = self._make_prompt(messages)
        response = self.llm_pipeline(composed_prompt, **self.model_kwargs)

        bt.logging.info(
            f"{self.__class__.__name__} generated the following output:\n{response}"
        )

        return response


if __name__ == "__main__":
    # Example usage
    llm_pipeline = vLLMPipeline(
        model_id="casperhansen/llama-3-70b-instruct-awq",
        device="cuda",
        llm_max_allowed_memory_in_gb=80,
        gpus=1,
    )
    llm = vLLM_LLM(llm_pipeline, system_prompt="You are a helpful AI assistant")

    message = "What is the capital of Texas?"
    response = llm.query(message)
    print(response)
