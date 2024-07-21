import random
from abc import ABC
from typing import List


class ChallengeTemplate(ABC):
    templates: List[str] = ["This is a template with {query}! <end>"]
    fields: dict = {"query": ["This is a placeholder for the query"]}

    def next(self, query: str) -> str:
        self.fields["query"] = [query]
        return (
            self.get_template()
            .format(
                **{
                    field: random.choice(entries)
                    for field, entries in self.fields.items()
                }
            )
            .split("<end>")[0]
        )

    def get_template(self) -> str:
        return random.choice(self.templates)
