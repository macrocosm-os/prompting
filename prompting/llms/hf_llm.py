import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, pipeline

from shared.settings import shared_settings


class ReproducibleHF:
    def __init__(self, model_id="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", **kwargs):
        """
        Initialize Hugging Face model with reproducible settings and optimizations
        """
        # Create a random seed for reproducibility
        # self.seed = random.randint(0, 1_000_000)
        # self.set_random_seeds(self.seed)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.valid_generation_params = set(
            AutoModelForCausalLM.from_pretrained(model_id).generation_config.to_dict().keys()
        )

        self.llm = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        self.sampling_params = shared_settings.SAMPLING_PARAMS

    @torch.inference_mode()
    def generate(self, messages: list[str] | list[dict], sampling_params=None, seed=None):
        """
        Generate text with optimized performance
        """
        self.set_random_seeds(seed)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(shared_settings.NEURON_DEVICE)

        params = sampling_params if sampling_params else self.sampling_params
        filtered_params = {k: v for k, v in params.items() if k in self.valid_generation_params}

        # Generate with optimized settings
        outputs = self.model.generate(
            **inputs.to(shared_settings.NEURON_DEVICE),
            **filtered_params,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        results = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0]

        # logger.debug(
        #     f"""{self.__class__.__name__} queried:
        #     prompt: {messages}\n
        #     responses: {results}\n
        #     sampling params: {params}\n
        #     seed: {seed}
        #     """
        # )

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


# if __name__ == "__main__":
#     llm = ReproducibleHF(model="Qwen/Qwen2-0.5B", tensor_parallel_size=1, seed=42)
#     llm.generate({"role": "user", "content": "Hello, world!"})
