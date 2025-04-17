from loguru import logger

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
except ImportError:
    logger.warning("Transformers or torch is not installed. This module will not be available.")

from .hf_llm import ReproducibleHF


class HFTextGeneration(ReproducibleHF):
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        device: str = "cuda:0",
        sampling_params: dict[str, str | float | int | bool] | None = None,
    ):
        super().__init__(model_id, device, sampling_params)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self._device,
        )
        self.model = self.model.to(self._device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.valid_generation_params = set(self.model.generation_config.to_dict().keys())

    @staticmethod
    def format_messages(messages: list[str] | list[dict[str, str]]) -> list[dict[str, str | list[dict[str, str]]]]:
        return messages
