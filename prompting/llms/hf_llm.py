import time
from typing import Optional, Any
from prompting.utils.cleaners import CleanerPipeline
from prompting.llms.base_llm import BaseLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig, pipeline
from loguru import logger
import random
import numpy as np
import torch
from prompting.utils.timer import Timer


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
    def __init__(self, model_id="Qwen/Qwen2-0.5B", tensor_parallel_size=0, seed=42, **kwargs):
        """
        Initialize Hugging Face model with reproducible settings and optimizations
        """
        self.set_random_seeds(seed)

        # Load model and tokenizer with optimizations
        model_kwargs = {
            "device_map": "auto",
        }

        # get valid params for generation from model config
        self.valid_generation_params = set(
            AutoModelForCausalLM.from_pretrained(model_id).generation_config.to_dict().keys()
        )

        for k, v in kwargs.items():
            if k not in ["sampling_params"]:  # exclude sampling_params and any other generation-only args
                model_kwargs[k] = v

        quantization_config = AwqConfig(
            bits=4,
            fuse_max_seq_len=512,
            do_fuse=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=quantization_config,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # self.model.generation_config.cache_implementation = "static"
        # self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)
        # self.valid_generation_params = set(self.model.generation_config.to_dict().keys())

        # Enable model optimizations
        self.model.eval()

        if tensor_parallel_size > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(tensor_parallel_size)))

        # Create pipeline with optimized settings
        self.llm = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        # Default sampling parameters
        self.sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_new_tokens": 256,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "seed": seed,
            "do_sample": True,
            "early_stopping": True,  # Enable early stopping
            "num_beams": 1,  # Use greedy decoding by default
        }

    @torch.inference_mode()
    def generate(self, prompts, sampling_params=None):
        """
        Generate text with optimized performance
        """

        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]

        inputs = self.tokenizer(prompts, truncation=True, return_tensors="pt").to(self.model.device)

        params = sampling_params if sampling_params else self.sampling_params
        filtered_params = {k: v for k, v in params.items() if k in self.valid_generation_params}

        with Timer() as timer:
            # Generate with optimized settings
            outputs = self.model.generate(
                **inputs,
                **filtered_params,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            results = [text.strip() for text in results]

        logger.debug(
            f"PROMPT: {prompts}\n\nRESPONSES: {results}\n\n"
            f"SAMPLING PARAMS: {params}\n\n"
            f"TIME FOR RESPONSE: {timer.elapsed_time}"
        )

        return results if len(results) > 1 else results[0]

    def set_random_seeds(self, seed=42):
        """
        Set random seeds for reproducibility across all relevant libraries
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    llm = ReproducibleHF(model="Qwen/Qwen2-0.5B", tensor_parallel_size=1, seed=42)
    llm.generate("Hello, world!")
