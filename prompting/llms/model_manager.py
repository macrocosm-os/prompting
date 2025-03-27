import asyncio
import gc
import multiprocessing as mp
import time
from typing import Optional

import psutil
import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr

from prompting.llms.hf_llm import ReproducibleHF
from prompting.llms.model_zoo import ModelConfig, ModelZoo
from prompting.llms.utils import GPUInfo, model_factory
from shared import settings
from shared.loop_runner import AsyncLoopRunner


class ModelManager(BaseModel):
    always_active_models: list[ModelConfig] = []
    total_ram: float = settings.shared_settings.LLM_MODEL_RAM
    active_models: dict[ModelConfig, ReproducibleHF] = {}
    used_ram: float = 0.0
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _mp_lock = PrivateAttr(default_factory=torch.multiprocessing.Lock)
    # TODO: Refactor it during refactor to get rid of global vars.
    event_restart: Optional[mp.Event] = None

    def load_always_active_models(self):
        for model_config in self.always_active_models:
            self.load_model(model_config=model_config)

    def load_model(self, model_config: ModelConfig, force: bool = True):
        """Load model into GPU.

        Warning: This operation will block execution until the model is successfully loaded into VRAM.

        Args:
            model_config: Model config to load.
            force: If enabled, will unload all other models.
        """
        logger.info(f"Loading {model_config.llm_model_id} model")
        try:
            with self._mp_lock:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
        except BaseException as e:
            logger.error(f"Error during CUDA empty cache: {e}")

        if model_config in self.active_models.keys():
            print(f"Model {model_config.llm_model_id} is already loaded.")
            return

        if force:
            logger.debug(f"Forcing model {model_config.llm_model_id} to load.")
            for active_model in list(self.active_models.keys()):
                if active_model in self.always_active_models:
                    continue
                logger.debug(f"Unloading {active_model.llm_model_id} to make room for {model_config.llm_model_id}")
                self._unload_model(active_model)

            self._vram_cleanup()

        retries_max = 3
        retry_counter = 0
        retry_delay = 30
        while True:
            try:
                GPUInfo.log_gpu_info()
                with self._mp_lock:
                    model = model_factory(model_config.llm_model_id)(
                        model_id=model_config.llm_model_id,
                        device=settings.shared_settings.NEURON_DEVICE,
                        sampling_params=settings.shared_settings.SAMPLING_PARAMS,
                    )
                    self.active_models[model_config] = model
                self.used_ram += model_config.min_ram
                logger.info(
                    f"Model {model_config.llm_model_id} has been successfully loaded. "
                    f"Approx. used VRAM: {self.used_ram:.0f}GB"
                )
                return model
            except BaseException as e:
                if retry_counter > retries_max:
                    logger.error(f"Failed to load model after {retries_max} retries. Terminating process")
                    # Fire an event to terminate the process.
                    self.event_restart.set()
                retry_counter += 1
                retry_delay += retry_counter
                self._vram_cleanup()
                logger.exception(
                    f"Failed to load model {model_config.llm_model_id}. Retrying in {retry_delay} seconds. "
                    f"Error: {str(e)}"
                )
                logger.debug(f"Current active models: {self.active_models}")
                time.sleep(retry_delay)

    def _cleanup_model(self, model_instance: ReproducibleHF, cpu_offload: bool = False):
        """Free VRAM from given model."""
        with self._mp_lock:
            if cpu_offload:
                try:
                    model_instance.model = model_instance.model.to("cpu")
                except NotImplementedError as e:
                    logger.exception(f"Standard move to CPU failed: {str(e)}")
                    try:
                        # Fallback for meta tensors.
                        model_instance.model = model_instance.model.to_empty("cpu")
                    except Exception as fallback_e:
                        logger.exception(
                            f"Could not move meta model to CPU, proceeding with generic GC: {str(fallback_e)}"
                        )
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

    def _unload_model(self, model_config: ModelConfig):
        if model_config not in self.active_models:
            logger.warning(f"Couldn't find given model to unload: {model_config}")
            return

        try:
            # Get the model instance
            model_instance = self.active_models[model_config]

            # Record initial memory state for debugging
            initial_free_memory = GPUInfo.free_memory
            logger.debug(f"Initial free GPU memory before unloading: {initial_free_memory} GB")

            # Check if system has enough RAM. Offloading model to CPU is more reliable to clean up VRAM.
            available_ram_gb = psutil.virtual_memory().available / 1024**3
            cpu_offload = available_ram_gb > model_config.min_ram
            if not cpu_offload:
                logger.warning(f"Cannot offload model to CPU, not enough RAM: {available_ram_gb:.2f} GB")
            self._cleanup_model(model_instance, cpu_offload=cpu_offload)

            # Remove the model from active models dictionary
            del self.active_models[model_config]

            self._vram_cleanup()

            # Report memory change.
            memory_freed = GPUInfo.free_memory - initial_free_memory
            logger.info(f"Successfully unloaded model {model_config.llm_model_id}. Memory freed: {memory_freed:.2f} GB")

        except Exception as ex:
            logger.error(f"Failed to unload model {model_config.llm_model_id}. Error: {str(ex)}")

        # Update used RAM tracking
        self.used_ram -= model_config.min_ram

        # Log current memory state
        GPUInfo.log_gpu_info()

    def get_model(self, llm_model: ModelConfig | str) -> ReproducibleHF:
        if not llm_model:
            llm_model = list(self.active_models.keys())[0] if self.active_models else ModelZoo.get_random()
        if isinstance(llm_model, str):
            llm_model = ModelZoo.get_model_by_id(llm_model)

        if llm_model in self.active_models:
            return self.active_models.get(llm_model)
        else:
            return self.load_model(llm_model, force=True)

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
        
        logger.info(f"Inferencing {dict_messages}")

        if isinstance(model, str):
            model = ModelZoo.get_model_by_id(model)
        if not model:
            model = ModelZoo.get_random(max_ram=self.total_ram)

        model_instance: ReproducibleHF = self.get_model(model)
        with self._mp_lock:
            responses = await model_instance.generate(
                messages=[dict_messages], sampling_params=sampling_params, seed=seed
            )

        return responses

    def _vram_cleanup(self):
        """Perform VRAM clean-up."""
        for config, model in self.active_models.items():
            del model.model
            del model

        self.active_models = {}
        self.used_ram = 0.0

        if torch.cuda.is_available():
            # Reset all CUDA cached memory.
            try:
                with self._mp_lock:
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                time.sleep(1.0)
            except BaseException as e:
                logger.error(f"Error during CUDA empty cache: {e}")
        else:
            logger.warning("CUDA is not available")

        gc.collect()
        gc.collect(generation=2)
        time.sleep(1.0)

        logger.info(f"VRAM clean-up completed. Current GPU usage: {GPUInfo.gpu_utilization * 100:.2f}%")
        GPUInfo.log_gpu_info()


class AsyncModelScheduler(AsyncLoopRunner):
    llm_model_manager: ModelManager
    interval: int = 14400
    scoring_queue: list | None = None

    async def start(self, scoring_queue: list, name: str | None = None, **kwargs):
        self.scoring_queue = scoring_queue
        return await super().start(name=name, **kwargs)

    async def initialise_loop(self):
        model_manager.load_always_active_models()

    async def run_step(self):
        """This method is called periodically according to the interval."""
        # try to load the model belonging to the oldest task in the queue
        selected_model = self.scoring_queue[0].task.llm_model if self.scoring_queue else None
        if not selected_model:
            selected_model = ModelZoo.get_random(max_ram=self.llm_model_manager.total_ram)
        logger.info(f"Loading model {selected_model.llm_model_id} for {self.interval} seconds.")

        if selected_model in self.llm_model_manager.active_models:
            logger.info(f"Model {selected_model.llm_model_id} is already loaded.")
            return

        logger.debug(f"Active models: {self.llm_model_manager.active_models.keys()}")
        # Load the selected model
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.llm_model_manager.load_model, selected_model)
        await asyncio.sleep(0.01)


model_manager = ModelManager()
model_scheduler = AsyncModelScheduler(llm_model_manager=model_manager, sync=True)
