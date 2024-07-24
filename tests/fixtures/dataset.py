from prompting.tools.datasets import (
    MockDataset,
    HFCodingDataset,
    WikiDataset,
    WikiDateDataset,
    MathDataset,
)

DATASETS = [
    MockDataset,
    HFCodingDataset,
    WikiDataset,
    WikiDateDataset,
    MathDataset,
]

wikidata = WikiDataset()

MOCK_CONTEXT = MockDataset().next()
WIKI_CONTEXT=wikidata.next(name="Emilio Alvarez (bishop)", method="get", selector = "all")
CODING_CONTEXT = HFCodingDataset(buffer_size=1, seed=42).next()
MATH_CONTEXT = MathDataset(seed=123).next()
DATEQA_CONTEXT = WikiDateDataset(seed=123).next()

print(WIKI_CONTEXT)

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
