import asyncio
import datetime
from abc import ABC, abstractmethod
from datetime import timedelta

import aiohttp
from loguru import logger
from pydantic import BaseModel, model_validator

from prompting.utils.profiling import profiler


class AsyncLoopRunner(BaseModel, ABC):
    interval: int = 10  # interval to run the main function in seconds
    running: bool = False
    sync: bool = False  # New parameter to enable/disable synchronization
    time_server_url: str = "http://worldtimeapi.org/api/ip"
    name: str | None = None
    step: int = 0

    @model_validator(mode="after")
    def validate_name(self):
        if self.name is None:
            self.name = self.__class__.__name__
        return self

    @abstractmethod
    async def run_step(self):
        """Implement this method with the logic that needs to run periodically."""
        raise NotImplementedError("run_step method must be implemented")

    async def get_time(self):
        """Get the current time from the time server with a timeout."""
        if not self.sync:
            time = datetime.datetime.now(datetime.timezone.utc)
            logger.debug(f"Time: {time}")
            return time
        try:
            async with aiohttp.ClientSession() as session:
                logger.info("Waiting for response time")
                async with session.get(self.time_server_url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info("Got response")
                        return datetime.datetime.fromisoformat(data["datetime"].replace("Z", "+00:00"))
                    else:
                        raise Exception(f"Failed to get server time. Status: {response.status}")
        except Exception as ex:
            logger.warning(f"Could not get time from server: {ex}. Falling back to local time.")
            return datetime.datetime.now(datetime.timezone.utc)

    def next_sync_point(self, current_time):
        """Calculate the next sync point based on the current time and interval."""
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        time_since_epoch = current_time - epoch
        seconds_since_epoch = time_since_epoch.total_seconds()
        next_interval = (seconds_since_epoch // self.interval + 1) * self.interval
        return epoch + timedelta(seconds=next_interval)

    async def wait_for_next_execution(self, last_run_time):
        """Wait until the next execution time, either synced or based on last run."""
        current_time = await self.get_time()
        logger.debug("Current time")
        if self.sync:
            next_run = self.next_sync_point(current_time)
        else:
            next_run = last_run_time + timedelta(seconds=self.interval)
        logger.debug(f"Next run: {next_run}")

        wait_time = (next_run - current_time).total_seconds()
        if wait_time > 0:
            logger.debug(
                f"{self.name}: Waiting for {wait_time:.2f} seconds until next {'sync point' if self.sync else 'execution'}"
            )
            await asyncio.sleep(wait_time)
        return next_run

    async def run_loop(self):
        """Run the loop periodically, optionally synchronizing across all instances."""
        logger.debug(f"Starting loop {self.__class__.__name__}; running: {self.running}")

        last_run_time = await self.get_time()
        logger.debug(f"Got time of last run: {last_run_time}")
        try:
            while self.running:
                with profiler.measure(self.name):
                    logger.debug("Waiting...")
                    next_run = await self.wait_for_next_execution(last_run_time)
                    logger.debug("Wait ended")
                    try:
                        await self.run_step()
                        self.step += 1
                        logger.debug(f"{self.name}: Step {self.step} completed at {next_run}")
                    except Exception as ex:
                        logger.exception(f"Error in loop iteration: {ex}")
                    last_run_time = next_run
        except asyncio.CancelledError:
            logger.info("Loop was stopped.")
        except Exception as e:
            logger.error(f"Fatal error in loop: {e}")
        finally:
            self.running = False
            logger.info("Loop has been cleaned up.")
        logger.debug("Exiting run_loop")

    async def start(self):
        """Start the loop."""
        if self.running:
            logger.warning("Loop is already running.")
            return
        self.running = True
        logger.debug(f"{self.name}: Starting loop with {'synchronized' if self.sync else 'non-synchronized'} mode")
        self._task = asyncio.create_task(self.run_loop())

    async def stop(self):
        """Stop the loop."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("Loop task was cancelled.")
