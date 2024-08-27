from loguru import logger
from pydantic import BaseModel, ConfigDict
import torch
import vllm
from prompting.llms.utils import GPUInfo
from vllm.distributed.parallel_state import destroy_model_parallel
from prompting.llms.model_zoo import ModelConfig, ModelZoo
from prompting.base.loop_runner import AsyncLoopRunner

# This maintains a list of tasks for which we need to generate references. Since
# we can only generate the references, when the correct model is loaded, we work
# through the tasks based on the currently loaded model.
open_tasks = []


class ModelManager(BaseModel):
    always_active_models: list[ModelConfig] = []
    total_ram: float = 40.0
    active_models: dict[ModelConfig, vllm.LLM] = {}
    used_ram: float = 0.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # @model_validator(mode="after")
    # def load_always_active_models(self) -> "ModelManager":
    #     for model_config in self.always_active_models:
    #         self.load_model(model_config)
    #     return self
    def load_always_active_models(self):
        for model_config in self.always_active_models:
            self.load_model(model_config)

    def load_model(self, model_config: ModelConfig, force: bool = True):
        if model_config in self.active_models.keys():
            print(f"Model {model_config.model_id} is already loaded.")
            return

        # if force loading is enabled, unload models until there is enough RAM
        if force:
            logger.debug(f"Forcing model {model_config.model_id} to load.")
            for active_model in list(self.active_models.keys()):
                logger.debug(f"Checking if model {active_model.model_id} can be unloaded.")
                if active_model in self.always_active_models:
                    logger.debug(f"Model {active_model.model_id} is always active. Skipping.")
                    continue
                if self.used_ram + model_config.min_ram > self.total_ram or GPUInfo.free_memory < model_config.min_ram:
                    logger.debug(f"Unloading {active_model.model_id} to make room for {model_config.model_id}")
                    self.unload_model(active_model)
                else:
                    logger.debug(f"Enough RAM for model {model_config.model_id} free")
                    GPUInfo.log_gpu_info()
                    break

        if self.used_ram + model_config.min_ram > self.total_ram or GPUInfo.free_memory < model_config.min_ram:
            if not force:
                logger.warning(f"Not enough RAM to load model {model_config.model_id}.")
                GPUInfo.log_gpu_info()
            raise MemoryError(
                f"""Not enough RAM to load model {model_config.model_id}.
                    Required: {model_config.min_ram} GB
                    Available in Model Manager: {self.total_ram - self.used_ram} GB
                    Available in GPU: {GPUInfo.free_memory} GB"""
            )

        try:
            model = vllm.LLM(
                model_config.model_id,
                max_model_len=8_000,
                gpu_memory_utilization=model_config.min_ram / GPUInfo.free_memory,
            )
            self.active_models[model_config] = model
            self.used_ram += model_config.min_ram
            logger.info(f"Model {model_config.model_id} loaded. Current used RAM: {self.used_ram} GB")

            return model
        except Exception as e:
            logger.exception(f"Failed to load model {model_config.model_id}. Error: {str(e)}")

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
            logger.error(f"Failed to unload model {model_config.model_id}. Error: {str(ex)}")
        gc.collect()
        self.used_ram -= model_config.min_ram
        torch.cuda.empty_cache()

    def get_or_load_model(self, model_id: str) -> vllm.LLM:
        model_config = ModelZoo.get_model_by_id(model_id)
        if model_config not in self.active_models:
            self.load_model(model_config)
        return self.active_models[model_config]

    def get_model(self, model: ModelConfig | str) -> vllm.LLM:
        if not model:
            model = list(self.active_models.keys())[0] if self.active_models else ModelZoo.get_random()
        if isinstance(model, str):
            model = ModelZoo.get_model_by_id(model)

        if model in self.active_models:
            return self.active_models.get(model)
        else:
            return self.load_model(model, force=True)


class AsyncModelScheduler(AsyncLoopRunner):
    model_manager: ModelManager
    interval: int = 10

    async def initialise_loop(self):
        model_manager.load_always_active_models()

    async def run_step(self):
        """This method is called periodically according to the interval."""
        # TODO: Make it load whichever model is in highest demand (based on the list of tasks) rather than a random model
        selected_model = ModelZoo.get_random(max_ram=self.model_manager.total_ram)
        logger.info(f"Loading model {selected_model.model_id} for {self.interval} seconds.")

        if selected_model in self.model_manager.active_models:
            logger.info(f"Model {selected_model.model_id} is already loaded.")
            return

        # Load the selected model
        model_manager.load_model(selected_model)


model_manager = ModelManager()
model_scheduler = AsyncModelScheduler(model_manager=model_manager)
