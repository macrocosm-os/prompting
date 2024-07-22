from abc import ABC, abstractmethod
from typing import Optional, Union
import bittensor as bt
import re
from typing import Union


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
        punctuation_chars: list[str] = [".", "?", "!"]

        if not any(char in generation for char in punctuation_chars):
            return generation

        if (
            not generation.endswith(".")
            and not generation.endswith("?")
            and not generation.endswith("!")
        ):
            index: int = max(generation.rfind(char) for char in punctuation_chars)
            return generation[
                : index + 1
            ]  # Go to the index of where the punctuation is, and include it (+1)
        else:
            return generation


class RemoveRoles(BaseCleaner):
    def __init__(self, **kwargs):
        pass

    def capitalize_sentences(self, input_string) -> str:
        """capitalize the first character after .!?"""
        sentences: list[str] = re.split(r"(?<=[.!?])\s+", input_string)
        capitalized_sentences: list[str] = [
            sentence.capitalize() for sentence in sentences
        ]
        result_string: str = " ".join(capitalized_sentences)
        # Capitalize the first letter in result_string
        result_string.capitalize()
        return result_string

    def apply(self, generation: str) -> str:
        generation: str = re.sub(r"\n*\w+\s*:", "", generation)
        roles: list[str] = [
            "User: ",
            "System: ",
            "Assistant: ",
            "Assistant, ",
            "Dear AI, ",
            "Dear AI ",
            "#Question: ",
            "<|im_start|>",
            "<|im_end|>",
            "<i>",
            "</i>",
        ]
        for role in roles:
            if role in generation:
                generation = generation.replace(role, "")

        return self.capitalize_sentences(
            input_string=generation
        )  # LLMs are good at being formal. Do the same if we remove a prefix.


class PrunePostQuestionText(BaseCleaner):
    def __init__(self, **kwargs):
        pass

    def apply(
        self,
        generation: str,
        min_pos: Union[int, float] = 5,
        max_pos: Union[int, float] = 0.5,
        max_questions: Optional[int] = None,
    ) -> str:

        if min_pos < 1:
            min_pos = int(min_pos * len(generation))
        if max_pos < 1:
            max_pos = int(max_pos * len(generation))

        # question mark occurs in first half of the query
        if not min_pos <= generation.rfind("?") <= max_pos:
            return generation
        elif max_questions is not None:
            generation: str = "?".join(generation.split("?", max_questions)[:-1]) + "?"
        else:
            # drop everything after the last question mark. Alternatively, we can just extract the first question.
            generation: str = generation.rsplit("?", 1) + "?"

        return generation


class RemoveTags(BaseCleaner):
    def __init__(self, **kwargs):
        pass

    def apply(self, generation: str) -> str:
        tags: list[str] = [
            "<date>",
        ]
        for tag in tags:
            if tag in generation:
                generation = generation.replace(tag, "")
        return generation


class FirstQuestion(BaseCleaner):
    def __init__(self, **kwargs):
        pass

    def apply(self, generation: str) -> str:
        if "?" in generation:
            if ":" in generation:
                generation = generation.split(":")[1]
            generation = generation.split("?")[0] + "?"
        return generation
