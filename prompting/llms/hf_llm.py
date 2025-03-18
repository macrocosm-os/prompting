import asyncio
import random
from functools import partial

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, pipeline


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
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self._device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.valid_generation_params = set(
            AutoModelForCausalLM.from_pretrained(model_id).generation_config.to_dict().keys()
        )
        self.llm = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    @torch.inference_mode()
    async def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        sampling_params: dict[str, str | float | int | bool] | None = None,
        seed: int | None = None,
    ) -> str:
        """Generate text with optimized performance asynchronously."""
        self.set_random_seeds(seed)

        # Move tokenization to a background thread since it can be CPU intensive
        loop = asyncio.get_event_loop()
        inputs = await loop.run_in_executor(
            None,
            partial(
                self.tokenizer.apply_chat_template,
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ),
        )
        inputs = inputs.to(self._device)

        params = sampling_params if sampling_params else self.sampling_params
        filtered_params = {k: v for k, v in params.items() if k in self.valid_generation_params}

        # Run model generation in a background thread to avoid blocking
        outputs = await loop.run_in_executor(
            None,
            partial(
                self.model.generate,
                **inputs,
                **filtered_params,
                eos_token_id=self.tokenizer.eos_token_id,
            ),
        )

        # Decode outputs in background thread
        results = await loop.run_in_executor(
            None,
            partial(
                self.tokenizer.batch_decode,
                outputs[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        )

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
