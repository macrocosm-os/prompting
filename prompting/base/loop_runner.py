import asyncio
from abc import ABC, abstractmethod
from loguru import logger
from pydantic import BaseModel, model_validator
from datetime import datetime, timedelta
import aiohttp
from prompting.utils.timer import Timer


class AsyncLoopRunner(BaseModel, ABC):
    interval: int = 10  # interval to run the main function in
    running: bool = False
    sync: bool = False  # New parameter to enable/disable synchronization
    _task: asyncio.Task = None
    time_server_url: str = "http://worldtimeapi.org/api/ip"  # Example time server
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

    async def initialise_loop(self):
        """Optional method to initialise any resources before starting the loop."""
        pass

    async def start(self):
        """Start the loop."""
        if self.running:
            logger.warning("Loop is already running.")
            return
        self.running = True
        logger.debug(f"{self.name}: Starting loop")
        await self.initialise_loop()
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

    async def wait_for_next_interval(self, last_run_time):
        """Wait until the start of the next interval."""
        if self.sync:
            current_time = await self.get_time()
        else:
            current_time = datetime.utcnow()

        next_run = last_run_time + timedelta(seconds=self.interval)
        next_run = next_run.replace(microsecond=0)  # Round to nearest second

        wait_time = (next_run - current_time).total_seconds()
        if wait_time > 0:
            logger.debug(f"{self.name}: Waiting for {wait_time} seconds")
            await asyncio.sleep(wait_time)
        return next_run

    async def run_loop(self):
        """Run the loop periodically, respecting the interval and optionally synchronizing with server time."""
        try:
            while self.running:
                try:
                    with Timer() as timer:
                        if self.sync:
                            try:
                                current_time = await self.get_time()
                            except Exception as e:
                                logger.warning(f"Failed to get server time: {e}. Falling back to local time.")
                                current_time = datetime.now(datetime.UTC)

                        await self.run_step()
                        self.step += 1

                    execution_time = timer.elapsed_time

                    if self.sync:
                        next_run = current_time.replace(microsecond=0) + timedelta(seconds=self.interval)
                        wait_time = (next_run - datetime.now(datetime.UTC)).total_seconds()
                        if wait_time > 0:
                            logger.debug(f"{self.name}: Waiting for {wait_time} seconds")
                            await asyncio.sleep(wait_time)
                    else:
                        remaining_time = self.interval - execution_time
                        if remaining_time > 0:
                            logger.debug(f"{self.name}: Waiting for {remaining_time} seconds")
                            await asyncio.sleep(remaining_time)

                except Exception as ex:
                    logger.exception(f"Error in loop iteration: {ex}")
                    await asyncio.sleep(self.interval)  # Wait before retrying
        except asyncio.CancelledError:
            logger.info("Loop was stopped.")
        except Exception as e:
            logger.error(f"Fatal error in loop: {e}")
        finally:
            self.running = False
            logger.info("Loop has been cleaned up.")

    async def get_time(self):
        """Get the current time from the time server with a timeout."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.time_server_url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return datetime.fromisoformat(data["datetime"].replace("Z", "+00:00"))
                    else:
                        raise Exception(f"Failed to get server time. Status: {response.status}")
        except Exception as ex:
            logger.warning(f"Could not get time from server: {ex}")
        return datetime.now(datetime.UTC)
