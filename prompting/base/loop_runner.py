import asyncio
from abc import ABC, abstractmethod
from loguru import logger
from pydantic import BaseModel


class AsyncLoopRunner(BaseModel, ABC):
    interval: int = 10  # interval to run the main function in
    running: bool = False
    _task: asyncio.Task = None
    step: int = 0

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

    async def run_loop(self):
        """Run the loop periodically, respecting the interval."""
        try:
            while self.running:
                try:
                    await self.run_step()
                    self.step += 1
                except Exception as ex:
                    logger.exception(ex)
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            logger.info("Loop was stopped.")
        except Exception as e:
            logger.error(f"Error in loop: {e}")
        finally:
            self.running = False
            logger.info("Loop has been cleaned up.")
