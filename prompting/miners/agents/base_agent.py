from abc import ABC

class BaseAgent(ABC):
    def run(self, input: str) -> str:
        pass