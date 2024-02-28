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

from transformers import Pipeline, pipeline
from prompting.mock import MockPipeline

from prompting.cleaners.cleaner import CleanerPipeline
import bittensor as bt
from prompting.mock import MockPipeline
from transformers import pipeline
from prompting.llms import BasePipeline, BaseLLM

        

def load_hf_pipeline(model_id, device=None, torch_dtype=None, mock=False, model_kwargs:dict = None):
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
            model_kwargs=model_kwargs
        )

    return llm_pipeline


class HuggingFacePipeline(BasePipeline):    
    def __init__(self, model_id, device=None, torch_dtype=None, mock=False, model_kwargs:dict = None):
        super().__init__()
        self.model = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        
        self.pipeline = load_hf_pipeline(model_id, device, torch_dtype, mock, model_kwargs)
        self.tokenizer = self.pipeline.tokenizer


    def __call__(self, system_prompt:str, prompt:str, **kwargs: dict) -> str:
        return self.pipeline(prompt, **kwargs)


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
        super().__init__(llm_pipeline, system_prompt, max_new_tokens, do_sample, temperature, top_k, top_p)
        
        self.kwargs = dict(
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_tokens,
        )

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

        if cleaner is not None:            
            clean_response = cleaner.apply(generation=response)
            if clean_response != response:
                bt.logging.debug(f"Response cleaned, chars removed: {len(response) - len(clean_response)}...")
            response = clean_response

        self.messages = messages + [{"content": response, "role": "assistant"}]
        self.times = self.times + [0, time.time() - tbeg]

        return response

    def __call__(self, messages: List[Dict[str, str]]):
        return self.forward(messages=messages)

    def _make_prompt(self, messages: List[Dict[str, str]]):
        return self.llm_pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def forward(self, messages: List[Dict[str, str]], preformat_messages: bool = False):
        composed_prompt = self._make_prompt(messages)
        # System prompt is composed in the prompt
        outputs = self.llm_pipeline(system_prompt=None, prompt=composed_prompt, **self.kwargs)
        response = outputs[0]["generated_text"]

        response = response.replace(composed_prompt, "").strip()

        bt.logging.info(
            f"{self.__class__.__name__} generated the following output:\n{response}"
        )
        return response
    