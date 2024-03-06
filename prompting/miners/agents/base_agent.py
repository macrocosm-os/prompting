from abc import ABC
from deprecation import deprecated


@deprecated(
    deprecated_in="1.1.2", removed_in="2.0", details="AgentMiner is unsupported."
)
class BaseAgent(ABC):
    def run(self, input: str) -> str:
        pass
