from .base import TemplateDataset


class GenericInstructionDataset(TemplateDataset):
    "Generic question dataset, which creates LLM prompts for asking questions."
    name = "generic_instruction"
    query_template = (
        "Ask a {style} question about a {theme} {subtopic} related to {topic}"
    )
    params = dict(
        style=[
            "casual",
            "basic",
            "silly",
            "random",
            "thoughtful",
            "detailed",
            "deep",
            "fun",
        ],
        theme=[
            "surprising",
            "controvesial",
            "historic",
            "modern",
            "famous",
            "imfamous",
            "popular",
            "unpopular",
        ],
        subtopic=[
            "person",
            "figure",
            "opinion",
            "event",
            "leader",
            "spokesperson",
            "expert",
            "topic",
        ],
        topic=[
            "science",
            "politics",
            "parenting",
            "travel",
            "cuisine",
            "sports",
            "pop culture",
            "tech",
            "history",
        ],
    )
