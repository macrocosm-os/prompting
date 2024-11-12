import time
from typing import Optional, Any
from prompting.utils.cleaners import CleanerPipeline
from prompting.llms.base_llm import BasePipeline, BaseLLM
from prompting.llms.utils import calculate_gpu_requirements
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import model_validator, ConfigDict
from loguru import logger
import random
import numpy as np
import torch
from prompting.utils.timer import Timer


def load_hf_pipeline(
    model_id: str,
    device: str,
    gpus: int,
    max_allowed_memory_in_gb: int,
    max_model_len: int,
    mock: bool = False,
    quantization: bool = True,
):
    """Loads the Hugging Face pipeline for the LLM, or a mock pipeline if mock=True"""

    max_allowed_memory_allocation_in_bytes = max_allowed_memory_in_gb * 1e9
    gpu_mem_utilization = calculate_gpu_requirements(device, gpus, max_allowed_memory_allocation_in_bytes)

    try:
        logger.info(
            f"Loading Hugging Face pipeline with model_id {model_id}: Max. VRAM: {gpu_mem_utilization}; GPUs: {gpus}"
        )
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        return hf_pipeline
    except Exception as e:
        logger.error(f"Error loading the Hugging Face pipeline within {max_allowed_memory_in_gb}GB: {e}")
        raise e


class HFPipeline(BasePipeline):
    llm_model_id: str
    llm_max_allowed_memory_in_gb: int
    llm_max_model_len: int
    mock: bool = False
    gpus: int = 1
    device: str = None
    quantization: bool = True
    llm: Optional[Any] = None
    tokenizer: Optional[AutoTokenizer] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def load_llm_and_tokenizer(self) -> "HFPipeline":
        self.llm = load_hf_pipeline(
            model_id=self.llm_model_id,
            device=self.device,
            gpus=self.gpus,
            max_allowed_memory_in_gb=self.llm_max_allowed_memory_in_gb,
            max_model_len=self.llm_max_model_len,
            mock=self.mock,
            quantization=self.quantization,
        )
        self.tokenizer = self.llm.tokenizer

    def __call__(self, composed_prompt: str, **model_kwargs: dict) -> str:
        if self.mock:
            return self.llm(composed_prompt, **model_kwargs)

        # Compose sampling params
        temperature = model_kwargs.get("temperature", 0.8)
        top_p = model_kwargs.get("top_p", 0.95)
        max_tokens = model_kwargs.get("max_tokens", 256)

        output = self.llm(composed_prompt, max_length=max_tokens, temperature=temperature, top_p=top_p)
        response = output[0]["generated_text"].strip()
        return response


class HF_LLM(BaseLLM):
    def __init__(
        self,
        llm: Any,
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
        message: list[str],
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
        response = self.llm.generate(
            composed_prompt,
            max_length=self.model_kwargs["max_tokens"],
            temperature=self.model_kwargs["temperature"],
            top_p=self.model_kwargs["top_p"],
        )[0]

        try:
            logger.info(
                f"{self.__class__.__name__} generated the following output:\n{response['generated_text'].strip()}"
            )
        except Exception as e:
            logger.info(f"Response: {response}")
            logger.error(f"Error logging the response: {e}")

        return response["generated_text"].strip()


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


class ReproducibleHF:
    def __init__(self, model="gpt2", tensor_parallel_size=1, seed=42, *args, **kwargs):
        """
        Initialize Hugging Face model with reproducible settings

        Args:
            model_name (str): HuggingFace model identifier
            tensor_parallel_size (int): Number of GPUs to use
            seed (int): Random seed for reproducibility
        """
        set_random_seeds(seed)

        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.llm = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=tensor_parallel_size)

        # Default sampling parameters for reproducibility
        self.sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_length": 400,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "seed": seed,
        }

    def generate(self, prompts, sampling_params=None):
        """
        Generate text with reproducible output

        Args:
            prompts: Single string or list of prompts
            sampling_params: Optional custom sampling parameters

        Returns:
            list: Generated outputs
        """
        with Timer() as timer:
            set_random_seeds(self.sampling_params["seed"])
            if isinstance(prompts, str):
                prompts = [prompts]

            # Use custom params if provided, else use default
            params = sampling_params if sampling_params else self.sampling_params

            # Generate
            outputs = self.llm(prompts, **params)

            # Extract generated text
            results = []
            for output in outputs:
                results.append(output["generated_text"].strip())
        logger.debug(
            f"PROMPT: {prompts}\n\nRESPONSES: {results}\n\nSAMPLING PARAMS: {sampling_params}\n\nTIME FOR RESPONSE: {timer.elapsed_time}"
        )

        return results if len(results) > 1 else results[0]
