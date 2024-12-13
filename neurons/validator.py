# ruff: noqa: E402
import asyncio
import multiprocessing as mp
import time

from loguru import logger

from prompting import mutable_globals, settings
from prompting.utils.profiling import profiler

settings.settings = settings.Settings.load(mode="validator")
settings = settings.settings

from prompting.api.api import start_api
from prompting.llms.model_manager import model_scheduler
from prompting.llms.utils import GPUInfo
from prompting.miner_availability.miner_availability import availability_checking_loop
from prompting.rewards.scoring import task_scorer
from prompting.tasks.task_creation import task_loop
from prompting.tasks.task_sending import task_sender
from prompting.weight_setting.weight_setter import weight_setter

NEURON_SAMPLE_SIZE = 100


async def main():
    # will start checking the availability of miners at regular intervals, needed for API and Validator
    asyncio.create_task(availability_checking_loop.start())

    if settings.DEPLOY_API:
        # Use multiprocessing to bypass API blocking issue.
        api_process = mp.Process(target=lambda: asyncio.run(start_api()))
        api_process.start()

    GPUInfo.log_gpu_info()
    if settings.DEPLOY_VALIDATOR:
        # start profiling
        asyncio.create_task(profiler.print_stats())

        # start rotating LLM models
        asyncio.create_task(model_scheduler.start())

        # start creating tasks
        asyncio.create_task(task_loop.start())

        # start sending tasks to miners
        asyncio.create_task(task_sender.start())

        # sets weights at regular intervals (synchronised between all validators)
        asyncio.create_task(weight_setter.start())

        # start scoring tasks in separate loop
        asyncio.create_task(task_scorer.start())

    # # TODO: Think about whether we want to store the task queue locally in case of a crash
    # # TODO: Possibly run task scorer & model scheduler with a lock so I don't unload a model whilst it's generating
    # # TODO: Make weight setting happen as specific intervals as we load/unload models
    start = time.time()
    while True:
        await asyncio.sleep(1)
        time_diff = -start + (start := time.time())
        logger.debug(f"Running {time_diff:.2f} seconds")


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    if settings.DEPLOY_API and settings.DEPLOY_VALIDATOR:
        # Replace scoring queue with memory-shared list between processes, before starting the loops.
        manager = mp.Manager()
        mutable_globals.scoring_queue = manager.list()

    asyncio.run(main())
