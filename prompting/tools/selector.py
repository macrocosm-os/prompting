import random


class Selector:
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = random.Random(seed)

    def __call__(self, items, weights=None):
        return self.rng.choices(items, weights=weights)[0]


class PageRankSelector(Selector):
    """Preferentially chooses the items at the top of the list, under the assumption that they are more important."""

    def __init__(self, seed=None, alpha=0.85):
        super().__init__(seed)
        self.alpha = alpha

    def __call__(self, items):
        weights = [self.alpha**i for i in range(len(items))]
        return self.rng.choices(items, weights=weights)[0]


class SimilaritySelector(Selector):
    """Chooses the item most similar to the query."""

    def __init__(self, seed=None, similarity_fn=None):
        super().__init__(seed)
        self.similarity_fn = similarity_fn

    def __call__(self, query, items):
        return max(items, key=lambda item: self.similarity_fn(query, item))


class TopSelector(Selector):
    """Chooses the top item."""

    def __init__(self, seed=None):
        super().__init__(seed)

    def __call__(self, items):
        return items[0]


if __name__ == "__main__":
    selector = Selector(seed=42)
    items = range(10)
    item = selector(items)

    assert item in items, "Selector should return one of the items"
