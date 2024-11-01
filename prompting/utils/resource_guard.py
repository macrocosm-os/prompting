from contextlib import asynccontextmanager
import asyncio

_global_async_lock = asyncio.Lock()


@asynccontextmanager
async def GuardResources():
    await _global_async_lock.acquire()
    try:
        yield
    finally:
        _global_async_lock.release()
