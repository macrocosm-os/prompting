from pydantic import BaseModel


class Context(BaseModel):
    # TODO: Pydantic model
    title: str
    topic: str
    subtopic: str
    content: str
    internal_links: list[str]
    external_links: list[str]
    source: str
    tags: list[str] = None
    extra: dict = None  # additional non-essential information
    stats: dict = None  # retrieval stats such as fetch time, number of tries, etc.
