from typing import Dict, List
import bittensor as bt
from prompting.cleaners import BaseCleaner


class RemoveQuotes(BaseCleaner):
    @property
    def name(self) -> str:
        return "remove_quotes"

    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, generation: str) -> str:
        bt.logging.debug("Pruning unfinished sentence.")
        return generation.strip("\"'")


class PruneEnding(BaseCleaner):
    @property
    def name(self) -> str:
        return "prune_ending"

    def __init__(self, **kwargs):
        pass

    def apply(self, generation: str) -> str:
        punctuation_chars = [".", "?", "!"]
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
    @property
    def name(self) -> str:
        return "remove_roles"

    def __init__(self, **kwargs):
        pass

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

        return (
            generation.capitalize()
        )  # LLMs are good at being formal. Do the same if we remove a prefix.
