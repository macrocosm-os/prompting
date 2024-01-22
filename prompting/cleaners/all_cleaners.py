from abc import ABC, abstractmethod
import bittensor as bt
import re


class BaseCleaner(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def apply(self, generation: str) -> str:
        pass


class RemoveQuotes(BaseCleaner):
    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, generation: str) -> str:
        bt.logging.debug("Pruning unfinished sentence.")
        return generation.strip("\"'")


class PruneEnding(BaseCleaner):
    def __init__(self, **kwargs):
        pass

    def apply(self, generation: str) -> str:
        punctuation_chars = [".", "?", "!"]

        if not any(char in generation for char in punctuation_chars):
            return generation

        if (
            not generation.endswith(".")
            and not generation.endswith("?")
            and not generation.endswith("!")
        ):
            index = max(generation.rfind(char) for char in punctuation_chars)
            return generation[
                : index + 1
            ]  # Go to the index of where the punctuation is, and include it (+1)
        else:
            return generation


class RemoveRoles(BaseCleaner):
    def __init__(self, **kwargs):
        pass

    def capitalize_sentences(self, input_string):
        """capitalize the first character after .!?"""
        sentences = re.split(r"(?<=[.!?])\s+", input_string)
        capitalized_sentences = [sentence.capitalize() for sentence in sentences]
        result_string = " ".join(capitalized_sentences)
        return result_string

    def apply(self, generation: str) -> str:
        roles = [
            "User: ",
            "System: ",
            "Assistant: ",
            "Assistant, ",
            "Dear AI, ",
            "Dear AI ",
            "#Question: ",
        ]
        for role in roles:
            if role in generation:
                generation = generation.replace(role, "")

        return self.capitalize_sentences(
            input_string=generation
        )  # LLMs are good at being formal. Do the same if we remove a prefix.
