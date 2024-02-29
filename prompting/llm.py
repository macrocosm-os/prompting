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

from transformers import Pipeline, pipeline, AutoTokenizer, TextIteratorStreamer
from prompting.mock import MockPipeline

from prompting.cleaners.cleaner import CleanerPipeline

# Some models don't use the same name for the tokenizer.
TOKENIZER_MAPPINGS = {"HuggingFaceH4/zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta"}


class CustomTextIteratorStreamer(TextIteratorStreamer):
    """
    TextIteratorStreamer stores print-ready text in a queue, to be used by a downstream application as an iterator.
    The queue is thread-safe and can be used to stream data from the model to the application.

    TextIteratorStreamer has internal methods to raise a StopIteration if a stop signal is received 
    (stop signal is when the value returned from the Queue is None), but this is not flexible enough. 
    Therefore, we add methods to check and clean the queue manually. 
    """

    def has_data(self):
        """Check if the queue has data."""
        return not self.text_queue.empty()

    def clear_queue(self):
        """Clear the queue."""
        with self.text_queue.mutex:  # ensures that the queue is cleared safely in a multi-threaded environment
            self.text_queue.queue.clear()


def load_pipeline(
    model_id: str,
    device=None,
    torch_dtype=None,
    mock=False,
    return_streamer=False,
    model_kwargs: dict = None,
):
    """Loads the HuggingFace pipeline for the LLM, or a mock pipeline if mock=True"""

    if mock or model_id == "mock":
        return MockPipeline(model_id)

    if not device.startswith("cuda"):
        bt.logging.warning("Only crazy people run this on CPU. It is not recommended.")

    tokenizer = (
        AutoTokenizer.from_pretrained(TOKENIZER_MAPPINGS[model_id])
        if model_id in TOKENIZER_MAPPINGS
        else None
    )
    streamer = CustomTextIteratorStreamer(tokenizer=tokenizer)

    # model_kwargs torch type definition conflicts with pipeline torch_dtype, so we need to differentiate them
    if model_kwargs is None:
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            streamer=streamer,
        )
    else:
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            device_map=device,
            model_kwargs=model_kwargs,
            streamer=streamer,
        )

    if not return_streamer:
        return llm_pipeline
    return llm_pipeline, streamer


class HuggingFaceLLM:
    def __init__(
        self,
        llm_pipeline: Pipeline,
        system_prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    ):
        self.llm_pipeline = llm_pipeline
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

    def query(
        self,
        message: List[Dict[str, str]],
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
                bt.logging.debug(
                    f"Response cleaned, chars removed: {len(response) - len(clean_response)}..."
                )
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
        prompt = self._make_prompt(messages)
        outputs = self.llm_pipeline(prompt, **self.kwargs)
        response = outputs[0]["generated_text"]

        response = response.replace(prompt, "").strip()

        bt.logging.info(
            f"{self.__class__.__name__} generated the following output:\n{response}"
        )
        return response
