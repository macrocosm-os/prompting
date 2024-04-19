from prompting.tools.datasets import (
    MockDataset,
    HFCodingDataset,
    WikiDataset,
    WikiDateDataset,
    MathDataset,
    BatchWikiDataset,
)

DATASETS = [
    MockDataset,
    HFCodingDataset,
    WikiDataset,
    WikiDateDataset,
    MathDataset,
]

BATCH_DATASETS = [
    BatchWikiDataset,
]

MOCK_CONTEXT = MockDataset().next()
WIKI_CONTEXT = WikiDataset().next(name="Emilio Alvarez (bishop)", method="get")
CODING_CONTEXT = HFCodingDataset(buffer_size=1, seed=42).next()
MATH_CONTEXT = MathDataset(seed=123).next()
DATEQA_CONTEXT = WikiDateDataset(seed=123).next()

CONTEXTS = {
    MockDataset: MOCK_CONTEXT,
    WikiDataset: WIKI_CONTEXT,
    HFCodingDataset: CODING_CONTEXT,
    MathDataset: MATH_CONTEXT,
    WikiDateDataset: DATEQA_CONTEXT,
}

CONTEXT_FIELDS = {
    "title": str,
    "topic": str,
    "subtopic": str,
    "content": str,
    "internal_links": list,
    "external_links": list,
    "source": str,
    "tags": list,
    "extra": dict,
    "stats": dict,
}
