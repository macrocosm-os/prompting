from loguru import logger
from pydantic import BaseModel, ConfigDict
import torch
import vllm
from transformers import pipeline, AutoTokenizer, Pipeline
import asyncio
from prompting.llms.utils import GPUInfo
from vllm.distributed.parallel_state import destroy_model_parallel
from prompting.llms.model_zoo import ModelConfig, ModelZoo
from prompting.base.loop_runner import AsyncLoopRunner
from prompting.mutable_globals import scoring_queue
from prompting.settings import settings
from vllm.sampling_params import SamplingParams
import random
import numpy as np
import os
import transformers

# This maintains a list of tasks for which we need to generate references. Since
# we can only generate the references, when the correct model is loaded, we work
# through the tasks based on the currently loaded model.
open_tasks = []


def set_seed(seed):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)


class ModelManager(BaseModel):
    always_active_models: list[ModelConfig] = []
    total_ram: float = settings.LLM_MODEL_RAM
    active_models: dict[ModelConfig, vllm.LLM] = {}
    used_ram: float = 0.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_always_active_models(self):
        for model_config in self.always_active_models:
            self.load_model(model_config)

    def load_model(self, model_config: ModelConfig, force: bool = True):
        if model_config in self.active_models.keys():
            print(f"Model {model_config.llm_model_id} is already loaded.")
            return

        # if force loading is enabled, unload models until there is enough RAM
        if force:
            logger.debug(f"Forcing model {model_config.llm_model_id} to load.")
            for active_model in list(self.active_models.keys()):
                logger.debug(f"Checking if model {active_model.llm_model_id} can be unloaded.")
                if active_model in self.always_active_models:
                    logger.debug(f"Model {active_model.llm_model_id} is always active. Skipping.")
                    continue
                if self.used_ram + model_config.min_ram > self.total_ram or GPUInfo.free_memory < model_config.min_ram:
                    logger.debug(f"Unloading {active_model.llm_model_id} to make room for {model_config.llm_model_id}")
                    self.unload_model(active_model)
                else:
                    logger.debug(f"Enough RAM for model {model_config.llm_model_id} free")
                    GPUInfo.log_gpu_info()
                    break

        if self.used_ram + model_config.min_ram > self.total_ram or GPUInfo.free_memory < model_config.min_ram:
            if not force:
                logger.warning(f"Not enough RAM to load model {model_config.llm_model_id}.")
                GPUInfo.log_gpu_info()
            raise MemoryError(
                f"""Not enough RAM to load model {model_config.llm_model_id}.
                    Required: {model_config.min_ram} GB
                    Available in Model Manager: {self.total_ram - self.used_ram} GB
                    Available in GPU: {GPUInfo.free_memory} GB"""
            )

        try:
            logger.debug(
                f"Loading model... {model_config.llm_model_id} with GPU Utilization: {model_config.min_ram / GPUInfo.free_memory}"
            )
            GPUInfo.log_gpu_info()
            pipe = pipeline("text-generation", model=settings.MINER_LLM_MODEL, max_length=1_000, device="cuda")
            self.active_models[model_config] = pipe
            self.used_ram += model_config.min_ram
            logger.info(f"Model {model_config.llm_model_id} loaded. Current used RAM: {self.used_ram} GB")
            return pipe

        except Exception as e:
            logger.exception(f"Failed to load model {model_config.llm_model_id}. Error: {str(e)}")

    def unload_model(self, model_config: ModelConfig):
        if model_config not in self.active_models:
            logger.warning("Couldn't find model to unload.")
            return
        import gc

        destroy_model_parallel()
        try:
            del self.active_models[model_config].llm_engine.model_executor.driver_worker
            del self.active_models[model_config]
        except Exception as ex:
            logger.error(f"Failed to unload model {model_config.llm_model_id}. Error: {str(ex)}")
        gc.collect()
        self.used_ram -= model_config.min_ram
        torch.cuda.empty_cache()

    def get_or_load_model(self, llm_model_id: str) -> Pipeline:
        model_config = ModelZoo.get_model_by_id(llm_model_id)
        if model_config not in self.active_models:
            self.load_model(model_config)
        return self.active_models[model_config]

    def get_model(self, llm_model: ModelConfig | str) -> Pipeline:
        if not llm_model:
            llm_model = list(self.active_models.keys())[0] if self.active_models else ModelZoo.get_random()
        if isinstance(llm_model, str):
            llm_model = ModelZoo.get_model_by_id(llm_model)

        if llm_model in self.active_models:
            return self.active_models.get(llm_model)
        else:
            return self.load_model(llm_model, force=True)

    # TODO: Merge generate and chat_generate into a single method
    def generate(
        self,
        prompt: str,
        model: ModelConfig | str | None = None,
        sampling_params: SamplingParams | None = SamplingParams(max_tokens=settings.NEURON_MAX_TOKENS),
    ) -> list[str]:
        if isinstance(model, str):
            model = ModelZoo.get_model_by_id(model)
        if not model:
            model = ModelZoo.get_random(max_ram=self.total_ram)

        model: Pipeline = self.get_model(model)
        # TODO: Load tokenizer only once
        set_seed(sampling_params.seed)
        responses = model(prompt)[0]["generated_text"]
        # response_results = [r.outputs[0].text.strip() for r in responses]
        logger.debug(f"PROMPT: \n\n{prompt}\n\nSAMPLING PARAMS: {sampling_params}\n\nOUTPUT:\n\n{responses}")
        return responses

    def chat_generate(
        self,
        messages: list[str],
        roles: list[str],
        model: ModelConfig | str | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> str:
        dict_messages = [{"content": message, "role": role} for message, role in zip(messages, roles)]
        tokenizer = AutoTokenizer.from_pretrained(settings.MINER_LLM_MODEL)
        composed_prompt = tokenizer.apply_chat_template(dict_messages, tokenize=False)
        logger.debug(f"Generating Chat with prompt: {composed_prompt}")
        return self.generate(composed_prompt, model=model, sampling_params=sampling_params)


class AsyncModelScheduler(AsyncLoopRunner):
    llm_model_manager: ModelManager
    interval: int = 14400

    async def initialise_loop(self):
        model_manager.load_always_active_models()

    async def run_step(self):
        """This method is called periodically according to the interval."""
        # try to load the model belonging to the oldest task in the queue
        selected_model = scoring_queue[0].task.llm_model if scoring_queue else None
        if not selected_model:
            selected_model = ModelZoo.get_random(max_ram=self.llm_model_manager.total_ram)
        logger.info(f"Loading model {selected_model.llm_model_id} for {self.interval} seconds.")

        if selected_model in self.llm_model_manager.active_models:
            logger.info(f"Model {selected_model.llm_model_id} is already loaded.")
            return

        logger.debug(f"Active models: {model_manager.active_models.keys()}")
        # Load the selected model
        model_manager.load_model(selected_model)
        await asyncio.sleep(0.01)


model_manager = ModelManager()
model_scheduler = AsyncModelScheduler(llm_model_manager=model_manager, sync=True)
