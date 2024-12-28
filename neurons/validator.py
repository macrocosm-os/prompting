import asyncio
import multiprocessing as mp
import time
import sys

import loguru
import torch

# ruff: noqa: E402
from shared import settings

shared_settings = settings.shared_settings
settings.shared_settings = settings.SharedSettings.load(mode="validator")


from prompting.llms.utils import GPUInfo


# Add a handler to write logs to a file
loguru.logger.add("logfile.log", rotation="1000 MB", retention="10 days", level="DEBUG")
from loguru import logger

torch.multiprocessing.set_start_method("spawn", force=True)

NEURON_SAMPLE_SIZE = 100


def create_loop_process(task_queue, scoring_queue, reward_events):
    async def spawn_loops(task_queue, scoring_queue, reward_events):
        # ruff: noqa: E402
        from shared import settings

        shared_settings = settings.shared_settings

        settings.shared_settings = settings.SharedSettings.load(mode="validator")

        from prompting.tasks.task_creation import task_loop
        from prompting.tasks.task_sending import task_sender
        from prompting.weight_setting.weight_setter import weight_setter
        from shared.profiling import profiler
        from prompting.rewards.scoring import task_scorer
        from prompting.miner_availability.miner_availability import availability_checking_loop
        from prompting.llms.model_manager import model_scheduler

        logger.info("Starting Profiler...")
        asyncio.create_task(profiler.print_stats(), name="Profiler"),

        # -------- Duplicate of create_task_loop ----------
        logger.info("Starting AvailabilityCheckingLoop...")
        asyncio.create_task(availability_checking_loop.start())

        logger.info("Starting TaskSender...")
        asyncio.create_task(task_sender.start(task_queue, scoring_queue))

        logger.info("Starting TaskLoop...")
        asyncio.create_task(task_loop.start(task_queue, scoring_queue))
        # -------------------------------------------------

        logger.info("Starting ModelScheduler...")
        asyncio.create_task(model_scheduler.start(scoring_queue), name="ModelScheduler"),
        logger.info("Starting TaskScorer...")
        asyncio.create_task(task_scorer.start(scoring_queue, reward_events), name="TaskScorer"),
        logger.info("Starting WeightSetter...")
        asyncio.create_task(weight_setter.start(reward_events))

        # Main monitoring loop
        start = time.time()

        logger.info("Starting Main Monitoring Loop...")
        while True:
            await asyncio.sleep(5)
            current_time = time.time()
            time_diff = current_time - start
            start = current_time

            # Check if all tasks are still running
            logger.debug(f"Running {time_diff:.2f} seconds")
            logger.debug(f"Number of tasks in Task Queue: {len(task_queue)}")
            logger.debug(f"Number of tasks in Scoring Queue: {len(scoring_queue)}")
            logger.debug(f"Number of tasks in Reward Events: {len(reward_events)}")

    asyncio.run(spawn_loops(task_queue, scoring_queue, reward_events))


def start_api():
    async def start():
        from prompting.api.api import start_scoring_api  # noqa: F401

        await start_scoring_api()
        while True:
            await asyncio.sleep(10)
            logger.debug("Running API...")

    asyncio.run(start())


# def create_task_loop(task_queue, scoring_queue):
#     async def start(task_queue, scoring_queue):
#         logger.info("Starting AvailabilityCheckingLoop...")
#         asyncio.create_task(availability_checking_loop.start())

#         logger.info("Starting TaskSender...")
#         asyncio.create_task(task_sender.start(task_queue, scoring_queue))

#         logger.info("Starting TaskLoop...")
#         asyncio.create_task(task_loop.start(task_queue, scoring_queue))
#         while True:
#             await asyncio.sleep(10)
#             logger.debug("Running task loop...")

#     asyncio.run(start(task_queue, scoring_queue))


async def main():
    # will start checking the availability of miners at regular intervals, needed for API and Validator
    with torch.multiprocessing.Manager() as manager:
        reward_events = manager.list()
        scoring_queue = manager.list()
        task_queue = manager.list()

        # Create process pool for managed processes
        processes = []

        try:
            # # Start checking the availability of miners at regular intervals

            if shared_settings.DEPLOY_SCORING_API:
                # Use multiprocessing to bypass API blocking issue
                api_process = mp.Process(target=start_api, name="API_Process")
                api_process.start()
                processes.append(api_process)

            loop_process = mp.Process(
                target=create_loop_process, args=(task_queue, scoring_queue, reward_events), name="LoopProcess"
            )
            # task_loop_process = mp.Process(
            #     target=create_task_loop, args=(task_queue, scoring_queue), name="TaskLoopProcess"
            # )
            loop_process.start()
            # task_loop_process.start()
            processes.append(loop_process)
            # processes.append(task_loop_process)
            GPUInfo.log_gpu_info()

            step = 0
            while True:
                await asyncio.sleep(30)
                if (
                    shared_settings.SUBTENSOR.get_current_block()
                    - shared_settings.METAGRAPH.last_update[shared_settings.UID]
                    > 500
                    and step > 120
                ):
                    logger.warning(
                        f"UPDATES HAVE STALED FOR {shared_settings.SUBTENSOR.get_current_block() - shared_settings.METAGRAPH.last_update[shared_settings.UID]} BLOCKS AND {step} STEPS"
                    )
                    logger.warning(
                        f"STALED: {shared_settings.SUBTENSOR.get_current_block()}, {shared_settings.METAGRAPH.block}"
                    )
                    sys.exit(1)
                step += 1

        except Exception as e:
            logger.error(f"Main loop error: {e}")
            raise
        finally:
            # Clean up processes
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
