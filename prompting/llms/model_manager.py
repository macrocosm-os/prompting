import asyncio
import gc
import multiprocessing as pymp
from typing import ClassVar

import torch
import torch.multiprocessing as mp
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from prompting.llms.vllm_llm import ReproducibleVLLM
from prompting.llms.model_zoo import ModelConfig, ModelZoo
from prompting.llms.utils import GPUInfo, model_factory
from shared import settings
from shared.loop_runner import AsyncLoopRunner


class ModelManager(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_restart: pymp.synchronize.Event = Field(default_factory=mp.Event)
    always_active_models: list[ModelConfig] = []
    total_ram: float = settings.shared_settings.LLM_MODEL_RAM
    active_models: dict[ModelConfig, ReproducibleVLLM] = {}
    used_ram: float = 0.0
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    async def load_always_active_models(self):
        for model_config in self.always_active_models:
            await self.load_model(model_config=model_config)

    async def load_model(self, model_config: ModelConfig, force: bool = True) -> ReproducibleHF:
        """Load model into GPU.

        Warning: This operation will block execution until the model is successfully loaded into VRAM.

        Args:
            model_config: Model config to load.
            force: If enabled, will unload all other models.
        """
        async with self._lock:
            if model_config in self.active_models.keys():
                logger.debug(f"Model {model_config.llm_model_id} is already loaded.")
                return self.active_models[model_config]

            if force:
                logger.debug(f"Forcing model {model_config.llm_model_id} to load.")
                for active_model in list(self.active_models.keys()):
                    if active_model in self.always_active_models:
                        continue
                    logger.debug(f"Unloading {active_model.llm_model_id} to make room for {model_config.llm_model_id}")

                    await self._unload_model(active_model)
                await self._vram_cleanup()

            retries_max = 1
            retry_counter = 0
            retry_delay = 15
            while True:
                try:
                    GPUInfo.log_gpu_info()
                    model = model_factory(model_config.llm_model_id)(
                        model_id=model_config.llm_model_id,
                        device=settings.shared_settings.NEURON_DEVICE,
                        sampling_params=settings.shared_settings.SAMPLING_PARAMS,
                    )
                    self.used_ram += model_config.min_ram
                    logger.info(
                        f"Model {model_config.llm_model_id} has been successfully loaded. "
                        f"Approx. used VRAM: {self.used_ram:.0f}GB"
                    )
                    self.active_models[model_config] = model
                    await asyncio.sleep(1.0)
                    return model
                except BaseException as e:
                    if retry_counter > retries_max:
                        logger.error(f"Failed to load model after {retries_max} retries. Terminating process")
                        await self._vram_cleanup()
                        # In case of VRAM leak, fire an event to terminate the process.
                        self.event_restart.set()
                        break

                    retry_counter += 1
                    retry_delay += retry_counter
                    await self._vram_cleanup()
                    logger.error(
                        f"Failed to load model {model_config.llm_model_id}. Retrying in {retry_delay} seconds. "
                        f"Error: {str(e)}"
                    )
                    logger.debug(f"Current active models: {self.active_models}")
                    await asyncio.sleep(retry_delay)

    async def _cleanup_model(self, model_instance: ReproducibleHF, cpu_offload: bool = False):
        """Free VRAM from given model."""
        if cpu_offload:
            try:
                model_instance.model = model_instance.model.to("cpu")
            except NotImplementedError as e:
                logger.exception(f"Standard move to CPU failed: {str(e)}")
                try:
                    # Fallback for meta tensors.
                    model_instance.model = model_instance.model.to_empty("cpu")
                except Exception as fallback_e:
                    logger.exception(f"Could not move meta model to CPU, proceeding with generic GC: {str(fallback_e)}")
            except Exception as e:
                logger.exception(f"Unexpected error when moving model to CPU: {str(e)}")

        model_instance.model = None
        model_instance.tokenizer = None
        try:
            del model_instance.model
            del model_instance.tokenizer
        except BaseException as e:
            logger.exception(f"Error deleting model attributes: {str(e)}")
        del model_instance

    async def _unload_model(self, model_config: ModelConfig):
        if model_config not in self.active_models:
            logger.warning(f"Couldn't find given model to unload: {model_config}")
            return

        try:
            model_instance = self.active_models.pop(model_config)

            # Record initial memory state for debugging.
            initial_free_memory = GPUInfo.free_memory
            logger.debug(f"Initial free GPU memory before unloading: {initial_free_memory} GB")

            await self._cleanup_model(model_instance, cpu_offload=False)
            await self._vram_cleanup()

            memory_freed = GPUInfo.free_memory - initial_free_memory
            logger.info(f"Successfully unloaded model {model_config.llm_model_id}. Memory freed: {memory_freed:.2f} GB")

        except Exception as ex:
            logger.error(f"Failed to unload model {model_config.llm_model_id}. Error: {str(ex)}")

        # Update used RAM tracking
        self.used_ram -= model_config.min_ram

        GPUInfo.log_gpu_info()

    async def get_model(self, llm_model: ModelConfig | str) -> ReproducibleHF:
        async with self._lock:
            if not llm_model:
                llm_model = list(self.active_models.keys())[0] if self.active_models else ModelZoo.get_random()
            if isinstance(llm_model, str):
                llm_model = ModelZoo.get_model_by_id(llm_model)
            if llm_model in self.active_models:
                return self.active_models[llm_model]

        return await self.load_model(llm_model, force=True)

    async def generate(
        self,
        messages: list[str],
        roles: list[str] | None = None,
        model: ModelConfig | str | None = None,
        seed: int = None,
        sampling_params: dict[str, float] = None,
    ) -> str:
        if messages and isinstance(messages[0], dict):
            dict_messages = messages
        else:
            dict_messages = [{"content": message, "role": role} for message, role in zip(messages, roles)]

        async with self._lock:
            if isinstance(model, str):
                model = ModelZoo.get_model_by_id(model)
            if not model:
                model = ModelZoo.get_random(max_ram=self.total_ram)

        model_instance: ReproducibleHF = await self.get_model(model)

        async with self._lock:
            if model_instance is None:
                raise ValueError("Model is None, which may indicate the model is still loading.")
            responses = await model_instance.generate(
                messages=[dict_messages], sampling_params=sampling_params, seed=seed
            )
            return responses

    async def _vram_cleanup(self):
        """Perform VRAM clean-up."""
        for _, model in self.active_models.items():
            del model.model
            del model

        self.active_models = {}
        self.used_ram = 0.0

        if torch.cuda.is_available():
            # Reset all CUDA cached memory.
            try:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                await asyncio.sleep(1.0)
            except BaseException as e:
                logger.error(f"Error during CUDA empty cache: {e}")
        else:
            logger.warning("CUDA is not available")

        gc.collect()
        gc.collect(generation=2)
        await asyncio.sleep(1.0)

        logger.info(f"VRAM clean-up completed. Current GPU usage: {GPUInfo.gpu_utilization * 100:.2f}%")
        GPUInfo.log_gpu_info()


class AsyncModelScheduler(AsyncLoopRunner):
    llm_model_manager: ModelManager
    interval: int = 10
    scoring_queue: list | None = None

    async def start(self, scoring_queue: list, name: str | None = None, **kwargs):
        self.scoring_queue = scoring_queue
        return await super().start(name=name, **kwargs)

    async def run_step(self):
        """This method is called periodically according to the interval."""
        if self.llm_model_manager.active_models:
            self.interval = 120 * 10
        # try to load the model belonging to the oldest task in the queue
        selected_model = self.scoring_queue[0].task.llm_model if self.scoring_queue else None
        if not selected_model:
            selected_model = ModelZoo.get_random(max_ram=self.llm_model_manager.total_ram)
        logger.info(f"Loading model {selected_model.llm_model_id} for {self.interval} seconds.")

        if selected_model in self.llm_model_manager.active_models:
            logger.info(f"Model {selected_model.llm_model_id} is already loaded.")
            return

        logger.debug(f"Active models: {self.llm_model_manager.active_models.keys()}")
        await self.llm_model_manager.load_model(selected_model)
        await asyncio.sleep(0.01)
