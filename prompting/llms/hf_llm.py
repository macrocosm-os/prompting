import random
from abc import abstractmethod

import numpy as np
from loguru import logger

try:
    import torch
except ImportError:
    logger.warning("torch is not installed. This module will not be available.")


class ReproducibleHF:
    def __init__(self, model_id: str, device: str, sampling_params: dict[str, str | float | int | bool] | None = None):
        self.model_id = model_id
        self._device = device
        self.sampling_params = sampling_params if sampling_params else {}

    @staticmethod
    @abstractmethod
    def format_messages(messages: list[str] | list[dict[str, str]]) -> list[dict[str, str | list[dict[str, str]]]]:
        raise NotImplementedError("This method must be implemented by the subclass")

    def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        sampling_params: dict[str, str | float | int | bool] | None = None,
        seed: int | None = None,
    ) -> str:
        """Generate text with optimized performance."""
        with torch.inference_mode():
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
