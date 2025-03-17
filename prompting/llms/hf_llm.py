import random

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)

from shared import settings

class ReproducibleHF:
    def __init__(
        self,
        model_id: str = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        device: str = "cuda:0",
        sampling_params: dict[str, str | float | int | bool] | None = None,
    ):
        """Deterministic HuggingFace model."""
        self._device = device
        self.sampling_params = {} if sampling_params is None else sampling_params
        if settings.shared_settings.LLM_TYPE.get(model_id) == "text-generation":
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=self._device,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.message_formater = lambda messages: messages
        elif settings.shared_settings.LLM_TYPE.get(model_id) == "image-text-to-text":
            self.model: AutoModelForImageTextToText = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=self._device,
            )
            self.tokenizer = AutoProcessor.from_pretrained(model_id)
            self.message_formater = self.gemma_format
        else:
            raise ValueError(f"Model type {settings.shared_settings.LLM_TYPE.get(model_id)} not supported")

        self.valid_generation_params = set(
            self.model.generation_config.to_dict().keys()
        )

    def gemma_format(self, messages: list[str] | list[dict[str, str]]) -> list[dict[str, str | list[dict[str, str]]]]:
        """Format the messages for the gemma model.
        
        Converts message content strings to dictionaries with type and text fields.
        Example:
        Input: [{"role": "user", "content": "Hello"}]
        Output: [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        """
        formatted_messages = []
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                # If content is a string, convert it to a list with a dictionary
                if isinstance(message["content"], str):
                    formatted_message = message.copy()
                    formatted_message["content"] = [{"type": "text", "text": message["content"]}]
                    formatted_messages.append(formatted_message)
                else:
                    # If content is already in the correct format, keep it as is
                    formatted_messages.append(message)
            else:
                # Handle other message formats if needed
                formatted_messages.append(message)
        
        return formatted_messages

    @torch.inference_mode()
    def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        sampling_params: dict[str, str | float | int | bool] | None = None,
        seed: int | None = None,
    ) -> str:
        """Generate text with optimized performance."""
        self.set_random_seeds(seed)

        inputs = self.tokenizer.apply_chat_template(
            self.message_formater(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._device)

        params = sampling_params if sampling_params else self.sampling_params
        filtered_params = {k: v for k, v in params.items() if k in self.valid_generation_params}

        outputs = self.model.generate(
            **inputs,
            **filtered_params,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        results = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0]

        return results if len(results) > 1 else results[0]

    def set_random_seeds(self, seed: int | None = 42):
        """Set random seeds for reproducibility across all relevant libraries."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
