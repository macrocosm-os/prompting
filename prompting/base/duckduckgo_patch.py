from typing import cast

import httpx
from threading import Event
from duckduckgo_search.duckduckgo_search import DDGS, DuckDuckGoSearchException, RatelimitException, TimeoutException


class PatchedDDGS(DDGS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = httpx.Client(
            headers=kwargs.get("headers", {}),
            proxy=kwargs.get("proxy", None),
            timeout=kwargs.get("timeout", 10),
            verify=kwargs.get("verify", True),
        )
        self._exception_event = Event()

    def _get_url(
        self: DDGS,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
        content: bytes | None = None,
        data: dict[str, str] | bytes | None = None,
    ) -> bytes:
        if self._exception_event.is_set():
            raise DuckDuckGoSearchException("Exception occurred in previous call.")
        try:
            resp = self.client.request(method, url, params=params, content=content, data=data)
        except Exception as ex:
            self._exception_event.set()
            if "time" in str(ex).lower():
                raise TimeoutException(f"{url} {type(ex).__name__}: {ex}") from ex
            raise DuckDuckGoSearchException(f"{url} {type(ex).__name__}: {ex}") from ex
        # logger.debug(f"_get_url() {resp.url} {resp.status_code} {len(resp.content)}") <--- Commented out as it yields error with httpx client
        if resp.status_code == 200:
            return cast(bytes, resp.content)
        self._exception_event.set()
        if resp.status_code in (202, 301, 403):
            raise RatelimitException(f"{resp.url} {resp.status_code} Ratelimit")
        raise DuckDuckGoSearchException(f"{resp.url} return None. {params=} {content=} {data=}")
