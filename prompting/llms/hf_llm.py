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
from typing import List, Dict
import bittensor as bt
from transformers import pipeline
from prompting.mock import MockPipeline
from prompting.cleaners.cleaner import CleanerPipeline
from transformers import pipeline
from prompting.llms import BasePipeline, BaseLLM


def load_hf_pipeline(
    model_id, device=None, torch_dtype=None, mock=False, model_kwargs: dict = None
):
    """Loads the HuggingFace pipeline for the LLM, or a mock pipeline if mock=True"""

    if mock or model_id == "mock":
        return MockPipeline(model_id)

    if not device.startswith("cuda"):
        bt.logging.warning("Only crazy people run this on CPU. It is not recommended.")

    # model_kwargs torch type definition conflicts with pipeline torch_dtype, so we need to differentiate them
    if model_kwargs is None:
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
    else:
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            device_map=device,
            model_kwargs=model_kwargs,
        )

    return llm_pipeline


class HuggingFacePipeline(BasePipeline):
    def __init__(
        self,
        model_id,
        device=None,
        torch_dtype=None,
        mock=False,
        model_kwargs: dict = None,
    ):
        super().__init__()
        self.model = model_id
        self.device = device
        self.torch_dtype = torch_dtype

        self.pipeline = load_hf_pipeline(
            model_id, device, torch_dtype, mock, model_kwargs
        )
        self.tokenizer = self.pipeline.tokenizer

    def __call__(self, composed_prompt: str, **kwargs: dict) -> str:
        return self.pipeline(composed_prompt, **kwargs)


class HuggingFaceLLM(BaseLLM):
    def __init__(
        self,
        llm_pipeline: BasePipeline,
        system_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    ):
        # Sets specific kwargs for hf pipeline
        model_kwargs = dict(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        super().__init__(llm_pipeline, system_prompt, model_kwargs)

        self.messages = [{"content": self.system_prompt, "role": "system"}]
        self.times = [0]

    def query(
        self,
        message: str,
        role: str = "user",
        disregard_system_prompt: bool = False,
        cleaner: CleanerPipeline = None,
    ):
        messages = self.messages + [{"content": message, "role": role}]

        if disregard_system_prompt:
            messages = messages[1:]

        tbeg = time.time()
        response = self.forward(messages=messages)
        response = self.clean_response(cleaner, response)

        self.messages = messages + [{"content": response, "role": "assistant"}]
        self.times = self.times + [0, time.time() - tbeg]

        return response

    def __call__(self, messages: List[Dict[str, str]]):
        return self.forward(messages=messages)

    def _make_prompt(self, messages: List[Dict[str, str]]):
        return self.llm_pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def forward(self, messages: List[Dict[str, str]]):
        composed_prompt = self._make_prompt(messages)
        # System prompt is composed in the prompt
        outputs = self.llm_pipeline(
            composed_prompt=composed_prompt, **self.model_kwargs
        )
        response = outputs[0]["generated_text"]

        response = response.replace(composed_prompt, "").strip()

        bt.logging.info(
            f"{self.__class__.__name__} generated the following output:\n{response}"
        )
        return response


if __name__ == "__main__":
    # Test the HuggingFacePipeline and HuggingFaceLLM
    model_id = "HuggingFaceH4/zephyr-7b-beta"
    device = "cuda"
    torch_dtype = "float16"
    mock = True

    llm_pipeline = HuggingFacePipeline(model_id, device, torch_dtype, mock)

    llm = HuggingFaceLLM(llm_pipeline, "You are a helpful AI assistant")

    message = "What is the capital of Texas?"
    response = llm.query(message)
    print(response)
