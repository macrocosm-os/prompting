import random
from typing import ClassVar

import datasets
import nltk
from nltk.corpus import wordnet
from pydantic import model_validator

from shared.base import BaseDataset, ChatEntry

nltk.download("wordnet")


class SN13Dataset(BaseDataset):
    _url: ClassVar[str] = "arrmlet/x_dataset_218"
    name: ClassVar[str] = "x_dataset_218"
    _chance_word_synonym: ClassVar[float] = 0.10
    _chance_char_typo: ClassVar[float] = 0.02
    exception: Exception | None = None
    dataset: datasets.Dataset = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def load_dataset(self) -> "SN13Dataset":
        self.dataset = datasets.load_dataset(self._url)["train"]
        self

    def get(self) -> ChatEntry:
        return self.sample()

    def next(self) -> ChatEntry:
        return self.sample()

    def random(self) -> ChatEntry:
        return self.sample()

    def sample(self) -> ChatEntry:
        """Sample the data, raises an exception if logging into HuggingFace was unsuccessful."""
        if self.exception is not None:
            raise self.exception
        # Randomly select a sample from the dataset.
        sample_idx = random.randint(0, len(self.dataset) - 1)
        message = self.dataset[sample_idx]["text"]
        role = ["user"]

        # Augment the messages by modifying words and introducing errors.
        messages = [self._augment_message(role, message)]

        return ChatEntry(roles=role, messages=messages, organic=False, source=self._url)

    def _augment_message(self, role: str, message: str) -> str:
        if role == "assistant":
            return message

        words = message.split()
        num_words_to_modify = random.randint(1, max(1, int(len(words) * self._chance_word_synonym)))
        words_to_modify = random.sample(range(len(words)), num_words_to_modify)

        for idx in words_to_modify:
            synonym = self._get_synonym(words[idx])
            if synonym:
                words[idx] = synonym

        message = " ".join(words)
        message = self._introduce_typos(message)
        return message

    def _get_synonym(self, word: str) -> str:
        synonyms = wordnet.synsets(word)
        if synonyms:
            # Choose a synonym that is not the word itself.
            synonym_words = [lemma.name() for lemma in synonyms[0].lemmas() if lemma.name() != word]
            if synonym_words:
                return random.choice(synonym_words)
        return word

    def _introduce_typos(self, message: str) -> str:
        message = list(message)
        num_errors = random.randint(0, max(1, int(len(message) * self._chance_char_typo)))
        for _ in range(num_errors):
            error_type = random.choice(["remove", "add_space"])
            error_position = random.randint(0, len(message) - 1)

            if error_type == "remove":
                message.pop(error_position)
            elif error_type == "add_space":
                message.insert(error_position, " ")

        return "".join(message)
