import time
from typing import Optional, Any
from prompting.utils.cleaners import CleanerPipeline
from prompting.llms.base_llm import BasePipeline, BaseLLM
from prompting.llms.utils import calculate_gpu_requirements
from vllm import LLM, RequestOutput
from transformers import PreTrainedTokenizerFast
from pydantic import model_validator, ConfigDict
from loguru import logger
from vllm import SamplingParams
import random
import numpy as np
import torch
from prompting.utils.timer import Timer

try:
    from vllm import SamplingParams
except ImportError:
    raise ImportError(
        "Could not import vllm library.  Please install via poetry: " 'poetry install --extras "validator" '
    )


def load_vllm_pipeline(
    model_id: str,
    device: str,
    gpus: int,
    max_allowed_memory_in_gb: int,
    max_model_len: int,
    mock: bool = False,
    quantization: bool = True,
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
        logger.info(f"Loading VLLM pipeline with model_id {model_id}: Max. VRAM: {gpu_mem_utilization}; GPUs: {gpus}")
        llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_mem_utilization,
            max_model_len=max_model_len,
            quantization="AWQ" if quantization else None,
            tensor_parallel_size=gpus,
            enforce_eager=True,
        )
        # This solution implemented by @bkb2135 sets the eos_token_id directly for efficiency in vLLM usage.
        # This approach avoids the overhead of loading a tokenizer each time the custom eos token is needed.
        # Using the Hugging Face pipeline, the eos token specific to llama models was fetched and saved (128009).
        # This method provides a straightforward solution, though there may be more optimal ways to manage custom tokens.
        llm.llm_engine.tokenizer.eos_token_id = 128009
        return llm
    except Exception as e:
        logger.error(f"Error loading the VLLM pipeline within {max_allowed_memory_in_gb}GB: {e}")
        raise e


class vLLMPipeline(BasePipeline):
    llm_model_id: str
    llm_max_allowed_memory_in_gb: int
    llm: LLM
    llm_max_model_len: int
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
            max_model_len=self.llm_max_model_len,
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
        response = output[0].outputs[0].text.strip()
        return response


class vLLM_LLM(BaseLLM):
    def __init__(
        self,
        llm: LLM,
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
        super().__init__(llm, system_prompt, model_kwargs)

        # Keep track of generation data using messages and times
        self.system_prompt = system_prompt
        self.messages = [{"content": self.system_prompt, "role": "system"}] if self.system_prompt else []
        self.times: list[float] = [0]
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
        """Query LLM with the given lists of conversation history and roles

        Args:
            messages (list[str]): List of messages in the conversation.
            roles (list[str]): List of roles for each message.
            cleaner (Optional[CleanerPipeline], optional): Cleaner pipeline to use, if any.
        """
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
        response: RequestOutput = self.llm.generate(composed_prompt, SamplingParams(**self.model_kwargs))[0]

        try:
            logger.info(
                f"{self.__class__.__name__} generated the following output:\n{response.outputs[0].text.strip()}"
            )
        except Exception as e:
            logger.info(f"Response: {response}")
            logger.error(f"Error logging the response: {e}")

        return response.outputs[0].text.strip()


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility across all relevant libraries
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ReproducibleVLLM:
    def __init__(self, model="meta-llama/Llama-2-7b-hf", tensor_parallel_size=1, seed=42, *args, **kwargs):
        """
        Initialize vLLM with reproducible settings

        Args:
            model_name (str): HuggingFace model identifier
            tensor_parallel_size (int): Number of GPUs to use
            seed (int): Random seed for reproducibility
        """
        set_random_seeds(seed)

        self.llm = LLM(
            model=model, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, seed=seed, *args, **kwargs
        )

        # Default sampling parameters for reproducibility
        self.sampling_params = SamplingParams(
            temperature=0.7, top_p=0.95, top_k=50, max_tokens=400, presence_penalty=0, frequency_penalty=0, seed=seed
        )

    def generate(self, prompts, sampling_params=None):
        """
        Generate text with reproducible output

        Args:
            prompts: Single string or list of prompts
            sampling_params: Optional custom SamplingParams
            seed: Optional seed override for this specific generation

        Returns:
            list: Generated outputs
        """
        with Timer() as timer:
            set_random_seeds(sampling_params.seed)
            if isinstance(prompts, str):
                prompts = [prompts]

            # Use custom params if provided, else use default
            params = sampling_params if sampling_params else self.sampling_params

            # Generate
            outputs = self.llm.generate(prompts, params)

            # Extract generated text
            results = []
            for output in outputs:
                results.append(output.outputs[0].text.strip())
        logger.debug(
            f"PROMPT: {prompts}\n\nRESPONSES: {results}\n\nSAMPLING PARAMS: {sampling_params}\n\nTIME FOR RESPONSE: {timer.final_time}"
        )

        return results if len(results) > 1 else results[0]
