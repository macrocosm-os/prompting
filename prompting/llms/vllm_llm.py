import time
import bittensor as bt
from typing import Optional
from prompting.utils.cleaners import CleanerPipeline
from prompting.llms.base_llm import BasePipeline, BaseLLM
from prompting.llms.utils import calculate_gpu_requirements
from vllm import LLM
from transformers import PreTrainedTokenizerFast
from pydantic import model_validator, ConfigDict

try:
    from vllm import SamplingParams
except ImportError:
    raise ImportError(
        "Could not import vllm library.  Please install via poetry: " 'poetry install --extras "validator" '
    )


def load_vllm_pipeline(
    model_id: str, device: str, gpus: int, max_allowed_memory_in_gb: int, mock: bool = False, quantization: bool = True
):
    """Loads the VLLM pipeline for the LLM, or a mock pipeline if mock=True"""

    try:
        from vllm import LLM
    except ImportError:
        raise ImportError(
            "Could not import vllm library.  Please install via poetry: " 'poetry install --extras "validator" '
        )

    # Calculates the gpu memory utilization required to run the model.
    max_allowed_memory_allocation_in_bytes = max_allowed_memory_in_gb * 1e9
    gpu_mem_utilization = calculate_gpu_requirements(device, gpus, max_allowed_memory_allocation_in_bytes)

    try:
        # Attempt to initialize the LLM
        llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_mem_utilization,
            quantization="AWQ" if quantization else None,
            tensor_parallel_size=gpus,
        )
        # This solution implemented by @bkb2135 sets the eos_token_id directly for efficiency in vLLM usage.
        # This approach avoids the overhead of loading a tokenizer each time the custom eos token is needed.
        # Using the Hugging Face pipeline, the eos token specific to llama models was fetched and saved (128009).
        # This method provides a straightforward solution, though there may be more optimal ways to manage custom tokens.
        llm.llm_engine.tokenizer.eos_token_id = 128009
        return llm
    except Exception as e:
        bt.logging.error(f"Error loading the VLLM pipeline within {max_allowed_memory_in_gb}GB: {e}")
        raise e


class vLLMPipeline(BasePipeline):
    llm_model_id: str
    llm_max_allowed_memory_in_gb: int
    mock: bool = False
    gpus: int = 1
    device: str = None
    quantization: bool = True

    llm: Optional[LLM] = None
    tokenizer: Optional[PreTrainedTokenizerFast] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def load_llm_and_tokenizer(self) -> "vLLMPipeline":
        self.llm = load_vllm_pipeline(
            model_id=self.llm_model_id,
            device=self.device,
            gpus=self.gpus,
            max_allowed_memory_in_gb=self.llm_max_allowed_memory_in_gb,
            mock=self.mock,
            quantization=self.quantization,
        )
        self.tokenizer = self.llm.llm_engine.get_tokenizer()

    def __call__(self, composed_prompt: str, **model_kwargs: dict) -> str:
        if self.mock:
            return self.llm(composed_prompt, **model_kwargs)

        # Compose sampling params
        temperature = model_kwargs.get("temperature", 0.8)
        top_p = model_kwargs.get("top_p", 0.95)
        max_tokens = model_kwargs.get("max_tokens", 256)

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
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
        self.times: list[float] = [0]
        self._role_template = {
            "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n{{{{ {} }}}}<|eot_id|>",
            "end": "<|start_header_id|>assistant<|end_header_id|>",
        }

    def query(
        self,
        message: str,
        role: str = "user",
        cleaner: CleanerPipeline = CleanerPipeline(),
    ):
        # Adds the message to the list of messages for tracking purposes, even though it's not used downstream
        messages = self.messages + [{"content": message, "role": role}]

        t0 = time.time()
        response = self._forward(messages=messages)
        response = self.clean_response(cleaner, response)

        self.messages = messages
        self.messages.append({"content": response, "role": "assistant"})
        self.times.extend((0, time.time() - t0))

        return response

    def _make_prompt(self, messages: list[dict[str, str]]) -> str:
        composed_prompt: list[str] = []

        for message in messages:
            role = message["role"]
            if role not in self._role_template:
                continue
            content = message["content"]
            composed_prompt.append(self._role_template[role].format(content))

        # Adds final tag indicating the assistant's turn
        composed_prompt.append(self._role_template["end"])
        return "".join(composed_prompt)

    def _forward(self, messages: list[dict[str, str]]):
        # make composed prompt from messages
        composed_prompt = self._make_prompt(messages)
        response = self.llm_pipeline(composed_prompt, **self.model_kwargs)

        bt.logging.info(f"{self.__class__.__name__} generated the following output:\n{response}")

        return response


if __name__ == "__main__":
    # Example usage
    llm_pipeline = vLLMPipeline(
        llm_model_id="casperhansen/llama-3-70b-instruct-awq",
        device="cuda",
        llm_max_allowed_memory_in_gb=80,
        gpus=1,
    )
    llm = vLLM_LLM(llm_pipeline, system_prompt="You are a helpful AI assistant")

    message = "What is the capital of Texas?"
    response = llm.query(message)
    print(response)
