import random

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
    def generate(
        self,
        messages: list[str] | list[dict[str, str]],
        sampling_params: dict[str, str | float | int | bool] | None = None,
        seed: int | None = None,
    ) -> str:
        """Generate text with optimized performance."""
        self.set_random_seeds(seed)

        inputs = self.tokenizer.apply_chat_template(
            messages,
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


# if __name__ == "__main__":
#     llm = ReproducibleHF(model="Qwen/Qwen2-0.5B", tensor_parallel_size=1, seed=42)
#     llm.generate({"role": "user", "content": "Hello, world!"})
