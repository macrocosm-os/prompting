import asyncio
import gc
import time

import psutil
import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict

from prompting.llms.hf_llm import ReproducibleHF
from prompting.llms.model_zoo import ModelConfig, ModelZoo
from prompting.llms.utils import GPUInfo, model_factory
from shared import settings
from shared.loop_runner import AsyncLoopRunner


class ModelManager(BaseModel):
    always_active_models: list[ModelConfig] = []
    total_ram: float = settings.shared_settings.LLM_MODEL_RAM
    active_models: dict[ModelConfig, ReproducibleHF] = {}
    model_usage_counters: dict[ModelConfig, int] = {}  # Track active generate calls per model
    model_locks: dict[ModelConfig, asyncio.Lock] = {}  # Locks for each model
    pending_unload: set[ModelConfig] = set()  # Models marked for unloading
    unload_tasks: dict[ModelConfig, asyncio.Task] = {}  # Tasks waiting to unload models
    used_ram: float = 0.0
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def load_always_active_models(self):
        for model_config in self.always_active_models:
            await self.load_model(model_config)

    async def load_model(self, model_config: ModelConfig, force: bool = True):
        """Load model into GPU.

        Warning: This operation will block execution until the model is successfully loaded into VRAM.

        Args:
            model_config: Model config to load.
            force: If enabled, will unload all other models.
        """
        logger.info(f"Loading {model_config.llm_model_id} model and unloading {self.active_models.keys()}")
        torch.cuda.empty_cache()
        if model_config in self.active_models.keys():
            if model_config in self.pending_unload:
                logger.info(f"Cancelling pending unload for model {model_config.llm_model_id}")
                if model_config in self.unload_tasks and not self.unload_tasks[model_config].done():
                    self.unload_tasks[model_config].cancel()
                self.pending_unload.remove(model_config)
            else:
                print(f"Model {model_config.llm_model_id} is already loaded.")
            return

        if force:
            logger.debug(f"Forcing model {model_config.llm_model_id} to load.")
            for active_model in list(self.active_models.keys()):
                if active_model in self.always_active_models:
                    continue

                # Request model unload (will happen after active calls complete)
                await self._request_unload(active_model)
                logger.info(f"Unloaded model {active_model.llm_model_id}")

            # Wait a short time to see if any models free up immediately
            if len(self.active_models) > 0:
                time.sleep(0.5)

            if len(self.active_models) == 0:
                await self._vram_cleanup()

        retries_max = 10
        retry_counter = 0
        retry_delay = 10
        while True:
            try:
                GPUInfo.log_gpu_info()
                model = model_factory(model_config.llm_model_id)(
                    model_id=model_config.llm_model_id,
                    device=settings.shared_settings.NEURON_DEVICE,
                    sampling_params=settings.shared_settings.SAMPLING_PARAMS,
                )
                self.active_models[model_config] = model
                # Initialize the lock and counter for this model
                if model_config not in self.model_locks:
                    self.model_locks[model_config] = asyncio.Lock()
                self.model_usage_counters[model_config] = 0
                self.used_ram += model_config.min_ram
                logger.info(
                    f"Model {model_config.llm_model_id} has been successfully loaded. "
                    f"Approx. used VRAM: {self.used_ram:.0f}GB"
                )
                return model
            except BaseException as e:
                if retry_counter > retries_max:
                    logger.error(f"Failed to load model after {retries_max} retries. Terminating process...")
                    # Terminate main process immediately by sending KeyboardInterrupt to all processes.
                    # TODO: Use sys.exit(1) instead and catch/propagate SystemExit in the main process.
                    import os
                    import signal

                    os.killpg(os.getpgid(os.getpid()), signal.SIGINT)
                retry_counter += 1
                retry_delay += retry_counter
                await self._vram_cleanup()
                logger.exception(
                    f"Failed to load model {model_config.llm_model_id}. Retrying in {retry_delay} seconds. "
                    f"Error: {str(e)}"
                )
                logger.debug(f"Current active models: {self.active_models}")
                time.sleep(retry_delay)

    async def _cleanup_model(self, model_instance: ReproducibleHF, cpu_offload: bool = False):
        """Free VRAM from given model."""
        logger.info(f"Cleaning up model {model_instance.model_id}")
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

    async def _wait_and_unload(self, model_config: ModelConfig):
        """Wait for all active calls to complete and then unload the model."""
        logger.info(
            f"Waiting for {self.model_usage_counters.get(model_config, 0)} active calls to complete for model {model_config.llm_model_id}"
        )

        # Wait until the usage counter reaches zero
        while self.model_usage_counters.get(model_config, 0) > 0:
            await asyncio.sleep(0.2)  # Check every 200ms

        logger.info(f"All active calls completed for model {model_config.llm_model_id}, proceeding with unload")

        # Perform the actual unload now that it's safe
        await self._perform_unload(model_config)

        # Remove from pending unload set
        if model_config in self.pending_unload:
            self.pending_unload.remove(model_config)

        # Remove the task
        if model_config in self.unload_tasks:
            del self.unload_tasks[model_config]

    async def _request_unload(self, model_config: ModelConfig):
        """Mark a model for unloading once all active calls complete."""
        logger.info(f"Requesting unload for model {model_config.llm_model_id}")
        if model_config not in self.active_models:
            logger.warning(f"Couldn't find model to mark for unloading: {model_config}")
            return

        if model_config in self.pending_unload:
            logger.info(f"Model {model_config.llm_model_id} is already marked for unloading")
            return

        # Add to pending unload set
        self.pending_unload.add(model_config)
        logger.info(f"Model {model_config.llm_model_id} marked for unloading when current calls complete")

        await self._wait_and_unload(model_config)

    async def _perform_unload(self, model_config: ModelConfig):
        """Actually unload the model from memory."""
        logger.info(f"Unloading model {model_config.llm_model_id}")
        if model_config not in self.active_models:
            logger.warning(f"Couldn't find model to unload: {model_config}")
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
            await self._cleanup_model(model_instance, cpu_offload=cpu_offload)

            # Remove the model from active models dictionary
            del self.active_models[model_config]

            # Clean up the locks and counters
            if model_config in self.model_locks:
                del self.model_locks[model_config]
            if model_config in self.model_usage_counters:
                del self.model_usage_counters[model_config]

            await self._vram_cleanup()

            # Report memory change.
            memory_freed = GPUInfo.free_memory - initial_free_memory
            logger.info(f"Successfully unloaded model {model_config.llm_model_id}. Memory freed: {memory_freed:.2f} GB")

        except Exception as ex:
            logger.error(f"Failed to unload model {model_config.llm_model_id}. Error: {str(ex)}")

        # Update used RAM tracking
        self.used_ram -= model_config.min_ram

        # Log current memory state
        GPUInfo.log_gpu_info()

    async def _unload_model(self, model_config: ModelConfig):
        """Request model unloading. Will be done after current calls complete."""
        # Instead of immediately unloading, mark for unloading and set up async task
        await self._request_unload(model_config)

    async def get_model(self, llm_model: ModelConfig | str) -> ReproducibleHF:
        if not llm_model:
            llm_model = list(self.active_models.keys())[0] if self.active_models else ModelZoo.get_random()
        if isinstance(llm_model, str):
            llm_model = ModelZoo.get_model_by_id(llm_model)

        # Check if the model is marked for unloading
        if llm_model in self.pending_unload:
            logger.info(f"Model {llm_model.llm_model_id} is pending unload, cannot use for new calls")
            # Find a different model or load a new one
            llm_model = ModelZoo.get_random(max_ram=self.total_ram)

        if llm_model in self.active_models:
            return self.active_models.get(llm_model)
        else:
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

        if isinstance(model, str):
            model = ModelZoo.get_model_by_id(model)
        if not model:
            model = ModelZoo.get_random(max_ram=self.total_ram)

        # If model is marked for unloading, choose a different one
        if model in self.pending_unload:
            logger.error(f"Model {model.llm_model_id} is pending unload, cannot use for this request")
            return None

            # model = ModelZoo.get_random(max_ram=self.total_ram)

        model_instance: ReproducibleHF = await self.get_model(model)

        # Mark model as in use
        if model not in self.model_locks:
            self.model_locks[model] = asyncio.Lock()

        try:
            # Increment the usage counter
            self.model_usage_counters[model] = self.model_usage_counters.get(model, 0) + 1
            logger.debug(f"Model {model.llm_model_id} usage count: {self.model_usage_counters[model]}")

            # Generate the response
            responses = await model_instance.generate(
                messages=[dict_messages], sampling_params=sampling_params, seed=seed
            )
            return responses
        finally:
            # Decrement the usage counter when done
            if model in self.model_usage_counters:
                self.model_usage_counters[model] -= 1
                logger.debug(
                    f"Model {model.llm_model_id} usage count after completion: {self.model_usage_counters[model]}"
                )

    async def _vram_cleanup(self):
        """Perform VRAM clean-up."""
        self.active_models = {}
        self.used_ram = 0.0

        # Reset model usage tracking
        self.model_usage_counters = {}
        self.model_locks = {}
        self.pending_unload = set()

        # Cancel any unload tasks
        for task in self.unload_tasks.values():
            if not task.done():
                task.cancel()
        self.unload_tasks = {}

        if torch.cuda.is_available():
            # Reset all CUDA cached memory.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            time.sleep(1.0)
        else:
            logger.warning("CUDA is not available")

        gc.collect()
        gc.collect(generation=2)
        time.sleep(1.0)

        logger.info(f"VRAM clean-up completed. Current GPU usage: {GPUInfo.gpu_utilization * 100:.2f}%")
        GPUInfo.log_gpu_info()


class AsyncModelScheduler(AsyncLoopRunner):
    llm_model_manager: ModelManager
    # interval: int = 14400
    interval: int = 10
    scoring_queue: list | None = None

    async def start(self, scoring_queue: list, name: str | None = None, **kwargs):
        self.scoring_queue = scoring_queue
        return await super().start(name=name, **kwargs)

    async def initialise_loop(self):
        await self.llm_model_manager.load_always_active_models()

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
        # await loop.run_in_executor(None, self.llm_model_manager.load_model, selected_model)
        await self.llm_model_manager.load_model(selected_model)
        await asyncio.sleep(0.01)


model_manager = ModelManager()
model_scheduler = AsyncModelScheduler(llm_model_manager=model_manager, sync=True)
