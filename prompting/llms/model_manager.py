import asyncio
import gc
import multiprocessing as pymp
from typing import ClassVar
import hashlib
import json
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from prompting.llms.model_zoo import ModelConfig, ModelZoo
from prompting.llms.utils import GPUInfo, model_factory
from prompting.llms.vllm_llm import ReproducibleVLLM
from shared import settings
from shared.loop_runner import AsyncLoopRunner


class AsyncRLock:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner = None
        self._count = 0

    async def acquire(self):
        current_task = asyncio.current_task()
        if self._owner == current_task:
            self._count += 1
            return True
        await self._lock.acquire()
        self._owner = current_task
        self._count = 1
        return True

    def release(self):
        current_task = asyncio.current_task()
        if self._owner != current_task:
            raise RuntimeError("Lock can only be released by the owner")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.release()


class ModelManager(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_restart: pymp.synchronize.Event = Field(default_factory=mp.Event)
    always_active_models: list[ModelConfig] = []
    total_ram: float = settings.shared_settings.LLM_MODEL_RAM
    active_models: dict[ModelConfig, ReproducibleVLLM] = {}
    used_ram: float = 0.0
    lock: ClassVar[AsyncRLock] = AsyncRLock()
    logits_cache: OrderedDict = Field(default_factory=OrderedDict)
    max_cache_size: int = 150 #Shouldn't need 150 generations per step, and we only need to cache per step

    async def load_always_active_models(self):
        for model_config in self.always_active_models:
            await self.load_model(model_config=model_config)

    async def load_model(self, model_config: ModelConfig, force: bool = True) -> ReproducibleVLLM:
        """Load model into GPU.

        Warning: This operation will block execution until the model is successfully loaded into VRAM.

        Args:
            model_config: Model config to load.
            force: If enabled, will unload all other models.
        """
        async with self.lock:
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

    async def _cleanup_model(self, model_instance: ReproducibleVLLM, cpu_offload: bool = False):
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

    async def get_model(self, llm_model: ModelConfig | str) -> ReproducibleVLLM:
        async with self.lock:
            if not llm_model:
                llm_model = list(self.active_models.keys())[0] if self.active_models else ModelZoo.get_random()
            if isinstance(llm_model, str):
                llm_model = ModelZoo.get_model_by_id(llm_model)
            if llm_model in self.active_models:
                return self.active_models[llm_model]

        return await self.load_model(llm_model, force=True)

    async def generate(
        self,
        messages: list[str] | list[dict],
        roles: list[str] | None = None,
        model: ModelConfig | str | None = None,
        seed: int = None,
        sampling_params: dict[str, float] = None,
    ) -> str:
        if messages and isinstance(messages[0], dict):
            dict_messages = messages
        else:
            dict_messages = [{"content": message, "role": role} for message, role in zip(messages, roles)]

        async with self.lock:
            if isinstance(model, str):
                model = ModelZoo.get_model_by_id(model)
            if not model:
                model = ModelZoo.get_random(max_ram=self.total_ram)

        model_instance: ReproducibleVLLM = await self.get_model(model)

        async with self.lock:
            if model_instance is None:
                raise ValueError("Model is None, which may indicate the model is still loading.")
            responses = await model_instance.generate(
                messages=dict_messages, sampling_params=sampling_params, seed=seed
            )
            return responses

    async def generate_logits(
        self,
        messages: list[str],
        model: ModelConfig | str | None = None,
        sampling_params: dict[str, float] = None,
        seed: int = None,
        continue_last_message: bool = False,
    ):
        # Create a hashable key for the cache
        if isinstance(model, ModelConfig):
            model_key = model.llm_model_id
        else: # If model is a string, it's a model ID
            model_key = model
            
        # Convert messages to a hashable format (tuple of strings)
        messages_key = tuple(messages)
        
        # Convert sampling_params to a hashable format
        sampling_params_key = json.dumps(sampling_params, sort_keys=True) if sampling_params else None
        
        # Create a cache key from all parameters
        cache_key = (messages_key, model_key, sampling_params_key, seed, continue_last_message)
        
        # Check if result is in cache
        if cache_key in self.logits_cache:
            logger.debug(f"Cache hit for logits generation with key {hash(cache_key)}")
            # Move this entry to the end to mark it as most recently used
            result = self.logits_cache.pop(cache_key)
            self.logits_cache[cache_key] = result
            return result
        
        # Not in cache, generate logits
        model_instance: ReproducibleVLLM = await self.get_model(model)
        result = await model_instance.generate_logits(
            messages=messages,
            sampling_params=sampling_params,
            seed=seed,
            continue_last_message=continue_last_message,
        )
        
        # Check if cache is at max capacity
        if len(self.logits_cache) >= self.max_cache_size:
            # Remove the oldest item (first item in OrderedDict)
            self.logits_cache.popitem(last=False)
            logger.debug(f"Cache limit reached, removed oldest entry. Cache size: {len(self.logits_cache)}")
        
        # Store in cache
        self.logits_cache[cache_key] = result
        logger.debug(f"Cached logits generation with key {hash(cache_key)}. Cache size: {len(self.logits_cache)}")
        
        return result

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
                logger.warning(f"Error during CUDA empty cache: {e}")
        else:
            logger.warning("CUDA is not available")

        gc.collect()
        gc.collect(generation=2)
        await asyncio.sleep(1.0)

        logger.info(f"VRAM clean-up completed. Current GPU usage: {GPUInfo.gpu_utilization * 100:.2f}%")
        GPUInfo.log_gpu_info()


class AsyncModelScheduler(AsyncLoopRunner):
    llm_model_manager: ModelManager
    interval: int = 1200
    scoring_queue: list | None = None

    async def start(self, scoring_queue: list, name: str | None = None, **kwargs):
        self.scoring_queue = scoring_queue
        await super().start(name=name, **kwargs)
        # Load the model immediately.
        await self.run_step()

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

        await self.llm_model_manager.load_model(selected_model)
        logger.debug(f"Active models: {self.llm_model_manager.active_models.keys()}")
        await asyncio.sleep(0.01)
