import asyncio

# import multiprocessing as mp
import torch.multiprocessing as mp
import sys

import loguru
import netaddr
import requests
import torch
import wandb
from bittensor.core.extrinsics.serving import serve_extrinsic

from prompting.rewards.scoring import task_scorer

# ruff: noqa: E402
from shared import settings
from shared.logging import init_wandb

settings.shared_settings = settings.SharedSettings.load(mode="validator")


from prompting.llms.utils import GPUInfo

# Add a handler to write logs to a file
loguru.logger.add("logfile.log", rotation="1000 MB", retention="10 days", level="DEBUG")
from loguru import logger

torch.multiprocessing.set_start_method("spawn", force=True)

NEURON_SAMPLE_SIZE = 100  # TODO: Should add this to constants.py


def create_loop_process(task_queue, scoring_queue, reward_events, miners_dict):
    settings.shared_settings = settings.SharedSettings.load(mode="validator")
    if settings.shared_settings.WANDB_ON:
        init_wandb(neuron="validator")

    async def spawn_loops(task_queue, scoring_queue, reward_events, miners_dict):
        # ruff: noqa: E402
        from prompting.llms.model_manager import model_scheduler

        # from prompting.miner_availability.miner_availability import availability_checking_loop
        from prompting.tasks.task_creation import task_loop
        from prompting.tasks.task_sending import task_sender
        from prompting.weight_setting.weight_setter import weight_setter
        from shared.profiling import profiler

        logger.info("Starting Profiler...")
        asyncio.create_task(profiler.print_stats(), name="Profiler"),

        # -------- Duplicate of create_task_loop ----------
        # logger.info("Starting AvailabilityCheckingLoop...")
        # asyncio.create_task(availability_checking_loop.start(miners_dict))

        logger.info("Starting TaskSender...")
        asyncio.create_task(task_sender.start(task_queue, scoring_queue, miners_dict))

        logger.info("Starting TaskLoop...")
        asyncio.create_task(task_loop.start(task_queue, scoring_queue, miners_dict, simultaneous_loops=1))
        # -------------------------------------------------

        logger.info("Starting ModelScheduler...")
        asyncio.create_task(model_scheduler.start(scoring_queue), name="ModelScheduler"),
        logger.info("Starting TaskScorer...")
        asyncio.create_task(task_scorer.start(scoring_queue, reward_events, simultaneous_loops=1), name="TaskScorer"),
        logger.info("Starting WeightSetter...")
        asyncio.create_task(weight_setter.start(reward_events))

        while True:
            await asyncio.sleep(5)

            # Check if all tasks are still running
            logger.debug(f"Number of tasks in Task Queue: {len(task_queue)}")
            logger.debug(f"Number of tasks in Scoring Queue: {len(scoring_queue)}")
            logger.debug(f"Number of tasks in Reward Events: {len(reward_events)}")

    try:
        asyncio.run(spawn_loops(task_queue, scoring_queue, reward_events, miners_dict))
    except Exception as e:
        logger.info(f"Terminating loop process: {e}")
    finally:
        logger.info("Cleaning up resources...")

        # Ensure wandb is closed properly
        if settings.shared_settings.WANDB_ON:
            wandb.finish()
            logger.info("WandB run finished.")


def start_api(scoring_queue, reward_events, miners_dict):
    async def start():
        from prompting.api.api import start_scoring_api  # noqa: F401

        # from prompting.miner_availability.miner_availability import availability_checking_loop

        # asyncio.create_task(availability_checking_loop.start(miners_dict))

        try:
            external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
            netaddr.IPAddress(external_ip)

            serve_success = serve_extrinsic(
                subtensor=settings.shared_settings.SUBTENSOR,
                wallet=settings.shared_settings.WALLET,
                ip=external_ip,
                port=settings.shared_settings.SCORING_API_PORT,
                protocol=4,
                netuid=settings.shared_settings.NETUID,
            )

            logger.debug(f"Serve success: {serve_success}")
        except Exception as e:
            logger.warning(f"Failed to serve scoring api to chain: {e}")
        await start_scoring_api(task_scorer, scoring_queue, reward_events, miners_dict)

        while True:
            await asyncio.sleep(10)

    asyncio.run(start())


def start_availability_checking_loop(miners_dict: dict):
    async def spawn_loops(miners_dict: dict):
        from prompting.miner_availability.miner_availability import availability_checking_loop

        logger.info("Starting availability checking loop in validator2...")
        asyncio.create_task(availability_checking_loop.start(miners_dict))
        while True:
            await asyncio.sleep(5)
            logger.debug("Availability checking loop is running")

    try:
        logger.info("Starting availability checking loop in validator...")
        asyncio.run(spawn_loops(miners_dict))

    except Exception as e:
        logger.exception(f"Availability checking loop error: {e}")
        raise


async def main():
    from prompting.tasks.task_registry import TaskRegistry
    from prompting.llms.model_zoo import ModelZoo

    # will start checking the availability of miners at regular intervals, needed for API and Validator
    with torch.multiprocessing.Manager() as manager:
        reward_events = manager.list()
        scoring_queue = manager.list()
        task_queue = manager.list()
        miners_dict = manager.dict()
        # mp_task_config = manager.dict(
        #     {str(task_config.task.__name__): True for task_config in TaskRegistry.task_configs}
        # )
        # mp_model_config = manager.dict({conf.llm_model_id: False for conf in ModelZoo.models_configs})

        # Create process pool for managed processes
        processes = []

        try:
            # Start checking the availability of miners at regular intervals
            if settings.shared_settings.DEPLOY_SCORING_API:
                # Use multiprocessing to bypass API blocking issue
                api_process = mp.Process(
                    target=start_api, args=(scoring_queue, reward_events, miners_dict), name="API_Process"
                )
                api_process.start()
                processes.append(api_process)

            availability_process = mp.Process(
                target=start_availability_checking_loop,
                args=(miners_dict,),
                name="AvailabilityProcess",
            )
            availability_process.start()
            processes.append(availability_process)

            loop_process = mp.Process(
                target=create_loop_process,
                args=(task_queue, scoring_queue, reward_events, miners_dict),
                name="LoopProcess",
            )
            loop_process.start()

            processes.append(loop_process)
            GPUInfo.log_gpu_info()

            step = 0
            while True:
                await asyncio.sleep(30)
                if (
                    settings.shared_settings.SUBTENSOR.get_current_block()
                    - settings.shared_settings.METAGRAPH.last_update[settings.shared_settings.UID]
                    > 500
                    and step > 120
                ):
                    current_block = settings.shared_settings.SUBTENSOR.get_current_block()
                    last_update_block = settings.shared_settings.METAGRAPH.last_update[settings.shared_settings.UID]
                    logger.warning(
                        f"Metagraph hasn't been updated for {current_block - last_update_block} blocks. "
                        f"Staled block: {current_block}, Last update: {last_update_block}"
                    )
                    break  # Exit the loop
                step += 1

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Shutting down gracefully...")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            raise
        finally:
            # Clean up processes
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
            sys.exit(1)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
