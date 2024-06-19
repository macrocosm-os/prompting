from typing import Any, Optional

from prompting.protocol import StreamPromptingSynapse
from prompting.tools.datasets.base import Dataset
import threading

from prompting.tools.selector import Selector


class OrganicDataset(Dataset):
    """Organic dataset singleton"""
    name = "organic"
    
    _instance = None
    _lock = threading.Lock()
    _queue: list[StreamPromptingSynapse] = []

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(OrganicDataset, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def add(cls, synapse: StreamPromptingSynapse):
        with cls._lock:
            cls._queue.append(synapse)

    @classmethod
    def random(cls, selector: Optional[Selector]) -> dict[str, Any]:
        with cls._lock:
            if cls._queue:
                synapse = cls._queue.pop(0)
                organic_source = "organic"
            else:
                # TODO: Get synthetic data.
                synapse = StreamPromptingSynapse(messages=["Synthetic organic: Capital of Australia?"], roles=["user"])
                organic_source = "synthetic"
        return {
            "title": "Prompt",
            "topic": "",
            "subtopic": "",
            "content": synapse.messages[-1],
            "internal_links": [],
            "external_links": [],
            "source": organic_source,
            "messages": synapse.messages,
            "roles": synapse.roles,
            "extra": {"date": None},
        }

    def get(self, name: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def search(self, name) -> dict[str, Any]:
        raise NotImplementedError


if __name__ == "__main__":
    dataset1 = OrganicDataset()
    dataset2 = OrganicDataset()
    dataset1.add(StreamPromptingSynapse(messages=["Capital of Australia?"], roles=["user"]))
    dataset1.add(StreamPromptingSynapse(messages=["Capital of South Korea?"], roles=["user"]))
    print(dataset2.random()["synapse"].messages)
    print(dataset2.random()["synapse"].messages)
    # This will raise an IndexError as the queue is empty
    # print(dataset2.random().messages) 