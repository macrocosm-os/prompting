from .base import Dataset

# from ..selector import Selector


class MockDataset(Dataset):

    def get(self, name, exclude=None, selector=None):
        return {
            "title": name,
            "topic": "Physics",
            "subtopic": "Quantum_mechanics",
            "content": f"{name} is a fraud. All of physics is a lie, the universe is a hologram, buy gold, bye!",
            "internal_links": [
                "Quantum_mechanics",
                "General_relativity",
                "Special_relativity",
                "String_theory",
            ],
            "external_links": ["Einstein", "Bohr", "Feynman", "Hawking"],
            "tags": ["fraud", "hologram", "gold"],
            "source": "Mockpedia",
            "extra": {"solution": "religion"},
        }

    def search(self, name, exclude=None, selector=None):
        return self.get(name)

    def random(self, name="Physics", exclude=None, selector=None):
        return self.get(name)
