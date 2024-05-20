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
from transformers import pipeline, TextIteratorStreamer, AutoTokenizer
from prompting.llms import BasePipeline, BaseLLM


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


def load_hf_pipeline(
    model_id: str,
    device=None,
    torch_dtype=None,
    mock=False,
    model_kwargs: dict = None,
    return_streamer: bool = False,
):
    """Loads the HuggingFace pipeline for the LLM, or a mock pipeline if mock=True"""

    if mock or model_id == "mock":
        return MockPipeline(model_id)

    if not device.startswith("cuda"):
        bt.logging.warning("Only crazy people run this on CPU. It is not recommended.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )  # model_id is usually the name of the tokenizer.
    except Exception as e:
        bt.logging.error(f"Failed to load tokenizer from model_id: {model_id}.")
        raise e

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

    if return_streamer:
        return llm_pipeline, streamer
    return llm_pipeline


class HuggingFacePipeline(BasePipeline):
    def __init__(
        self,
        model_id,
        device=None,
        torch_dtype=None,
        mock=False,
        model_kwargs: dict = None,
        return_streamer: bool = False,
        gpus: int = 1,
        llm_max_allowed_memory_in_gb: int = 0
    ):
        super().__init__()
        self.model = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.mock = mock

        package = load_hf_pipeline(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            mock=mock,
            model_kwargs=model_kwargs,
            return_streamer=return_streamer,
        )

        if return_streamer:
            self.pipeline, self.streamer = package
        else:
            self.pipeline = package

        self.tokenizer = self.pipeline.tokenizer

    def __call__(self, composed_prompt: str, **kwargs: dict) -> str:
        if self.mock:
            return self.pipeline(composed_prompt, **kwargs)

        # Extract the generated text from the pipeline output
        outputs = self.pipeline(composed_prompt, **kwargs)
        return outputs[0]["generated_text"]


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

    def stream(
        self,
        message: str,
        role: str = "user",
    ):
        messages = self.messages + [{"content": message, "role": role}]
        prompt = self._make_prompt(messages)

        bt.logging.debug("Starting LLM streaming process...")
        streamer = CustomTextIteratorStreamer(tokenizer=self.llm_pipeline.tokenizer)
        _ = self.llm_pipeline(prompt, streamer=streamer, **self.model_kwargs)

        return streamer

    def __call__(self, messages: List[Dict[str, str]]):
        return self.forward(messages=messages)

    def _make_prompt(self, messages: List[Dict[str, str]]):
        return self.llm_pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def forward(self, messages: List[Dict[str, str]]):
        composed_prompt = self._make_prompt(messages)
        # System prompt is composed in the prompt
        response = self.llm_pipeline(
            composed_prompt=composed_prompt, **self.model_kwargs
        )

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

    llm_pipeline = HuggingFacePipeline(
        model_id=model_id, device=device, torch_dtype=torch_dtype, mock=mock
    )

    llm = HuggingFaceLLM(llm_pipeline, "You are a helpful AI assistant")

    message = "What is the capital of Texas?"
    response = llm.query(message)
    print(response)
