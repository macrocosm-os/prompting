import re
import sys
import random
import bittensor as bt
import wikipedia
from queue import Queue, Full, Empty
from functools import lru_cache
from prompting.datasets.base import BaseDataset
from prompting.datasets.base import Context
from typing import ClassVar
from typing import Optional
from pydantic import model_validator, ConfigDict

# Create a queue called CACHED_ARTICLES to store wikipedia articles that have been fetched
CACHED_ARTICLES: Queue[Context] = Queue(maxsize=300)


# speed up page loading
@lru_cache(maxsize=1000)
def _get_page(title, pageid=None, auto_suggest=False, redirect=True, seed=None) -> wikipedia.WikipediaPage:
    """Cached Wikipedia page loading."""
    try:
        page = wikipedia.page(title=title, pageid=pageid, auto_suggest=auto_suggest, redirect=redirect)
        # create sections manually if not found
        if not page.sections:
            page._sections = [
                line.strip("= ") for line in page.content.splitlines() if re.search(r"=+\s+.*\s+=+", line)
            ]
        return page

    except wikipedia.DisambiguationError as e:
        bt.logging.debug(f"{e.__class__.__name__} loading page {title!r}: {e}")
        # exc info contains a tuple of (requested_title: str, possible_matches: list[str])
        pages = sys.exc_info()[1].args[1]
        if not isinstance(pages, list):
            return None
        title = random.Random(seed).choice(pages)
        return _get_page(title, auto_suggest=auto_suggest, redirect=redirect)

    except wikipedia.PageError as e:
        bt.logging.warning(f"{e.__class__.__name__} loading page {title!r}: {e}")
        if not auto_suggest:
            return _get_page(title, auto_suggest=True, redirect=redirect)
        return None


@lru_cache(maxsize=1000)
def _get_random_titles(pages=10, seed=42) -> list:
    """Cached wikipedia random page. Approximately deterministic random titles. This is useful for testing.
    NOTE: the actually cached result will change each session, but the result will be the same within a session.
    """
    return wikipedia.random(pages=pages)


@lru_cache(maxsize=1000)
def _wikipedia_search(name, results) -> list:
    """Cached Wikipedia search."""
    return wikipedia.search(name, results=results)


def process_page(page, valid_header: callable = None, valid_content: callable = None) -> dict:
    """Process a Wikipedia page and return a dictionary of sections with their content.

    Args:
        page: wikipedia.WikipediaPage
        valid_header: callable to determine if a section header is valid
        valid_content: callable to determine if a section content is valid
    Returns:
        dict: dictionary of sections and their content. Note that keys are tuples (header, section_title)
    """
    header = ""
    sections = {}

    for section_title in page.sections:
        content = page.section(section_title)
        if not content:
            header = section_title
            continue

        # Filter out sections that don't match the headers and/or are not valid
        if (valid_header and not valid_header(header)) or (valid_content and not valid_content(content)):
            continue

        key = (header, section_title)
        sections[key] = content.splitlines()

    if not sections:
        bt.logging.debug(f"No valid sections found in page {page.title!r} ({page.url})")

    return sections


def most_relevant_links(page, num_links=10, num_summary_words=50, return_scores=False):
    """Return the most relevant links to a Wikipedia page based on the intersection over union (IOU) of the link and the page summary."""
    link_scores = {}
    summary_words = set(page.summary.split()[:num_summary_words])
    for link in page.links:
        link_words = set(link.split())
        iou = len(summary_words.intersection(link_words)) / len(summary_words.union(link_words))
        link_scores[link] = iou / len(link.split())

    sorted_links = sorted(link_scores.items(), key=lambda x: x[1], reverse=True)
    if return_scores:
        return sorted_links[:num_links]

    return [link for link, _ in sorted_links[:num_links]]


def filter_categories(categories, exclude=None, include=None):
    """Filter categories based on a list of categories to exclude and/or include."""
    if exclude:
        categories = [cat for cat in categories if not re.search("|".join(exclude), cat, re.IGNORECASE)]
    if include:
        categories = [cat for cat in categories if re.search("|".join(include), cat, re.IGNORECASE)]
    return categories


class WikiDataset(BaseDataset):
    """Wikipedia dataset. Uses the wikipedia python api to fetch articles and sections."""

    name: ClassVar[str] = "wikipedia"
    EXCLUDE_HEADERS: tuple = ("See also", "References", "Further reading", "External links")
    EXCLUDE_CATEGORIES: tuple = ("articles", "wikipedia", "pages", "cs1")
    min_length_words: int = 50
    max_links: int = 10

    def get(
        self,
        name: str,
        include: list = None,
        exclude: list = None,
        **kwargs,
    ) -> Context:
        """Get a specified Wikipedia page and extract a section based on the selector.

        Args:
            name (_type_): _description_
            pageid (_type_, optional): _description_. Defaults to None.
            auto_suggest (bool, optional): _description_. Defaults to True.
            redirect (bool, optional): _description_. Defaults to True.
            selector (Selector, optional): _description_. Defaults to None.
            include (list, optional): _description_. Defaults to None.
            exclude (list, optional): _description_. Defaults to None.

        Returns:
            dict: _description_
        """

        page = _get_page(title=name, **kwargs)
        if page is None:
            return None

        # Only return a sections with a minimum number of words
        exclude = (exclude or []) + list(self.EXCLUDE_HEADERS)
        sections = process_page(
            page,
            valid_header=lambda x: x not in exclude and (not include or x in include),
            valid_content=lambda x: len(x.split()) >= self.min_length_words,
        )
        if not sections:
            return None

        key = header, section_title = random.choice(list(sections.keys()))
        content = "\n".join(sections[key])
        section_length = len(content.split())

        context = Context(
            title=name,
            topic=header or section_title,
            subtopic=section_title,
            content=section_title,
            internal_links=list(filter(lambda x: x not in exclude, page.sections)),
            external_links=most_relevant_links(page, num_links=self.max_links),
            tags=filter_categories(page.categories, exclude=self.EXCLUDE_CATEGORIES),
            source="Wikipedia",
            extra={
                "url": page.url,
                "page_length": len(page.content.split()),
                "section_length": section_length,
            },
        )
        try:
            CACHED_ARTICLES.put(context, block=False)
        except Full:
            bt.logging.debug("Cache is full. Skipping article until cache is emptied.")
        return context

    def search(self, name, results=3) -> Context:
        titles = _wikipedia_search(name, results=results)
        title = random.choice(titles)
        return self.get(title)

    def random(self, pages=10, seed=None) -> Context:
        titles = wikipedia.random(pages=pages) if seed is None else _get_random_titles(pages=pages, seed=seed)
        title = random.choice(titles)
        return self.get(title)


class WikiDateDataset(BaseDataset):
    name: str = "wikipedia_date"
    INCLUDE_HEADERS: tuple = ("Events", "Births", "Deaths")
    MONTHS: tuple = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    EXCLUDE_CATEGORIES: tuple = ("articles", "wikipedia", "pages", "cs1")
    max_tries: int = 10
    seed: int | None = None
    rng: Optional[random.Random] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def create_rng(self) -> "WikiDateDataset":
        self.rng = random.Random(self.seed)
        return self

    def _extract_dates_and_sentences(self, text: str) -> tuple[str, str]:
        # Regular expression to find dates in various formats
        date_pattern = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember))\s+\d{4}\b|\b\d{4}\b"

        # Compile the regex pattern
        date_regex = re.compile(date_pattern)

        # Split text into sentences
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

        # Iterate through sentences and find dates
        for sentence in sentences:
            # Find all dates in the sentence
            dates = date_regex.findall(sentence)
            # If dates are found, add them to the result dictionary with the corresponding sentence
            if dates:
                for date in dates:
                    # Return the first date found
                    return (str(date), sentence.replace(str(date), "<date>").strip())
        return None

    def _random_date(self) -> str:
        for _ in range(self.max_tries):
            try:
                context = CACHED_ARTICLES.get(block=False)
                if not context:
                    continue

                date_sentence = self._extract_dates_and_sentences(context.content)
                if not date_sentence:
                    continue

                context.content, context.extra["date"] = date_sentence[1], date_sentence[0]
                return context

            except Empty:
                bt.logging.debug("Cache is empty. Skipping date until cache is filled.")
                return None

    def get(
        self,
    ) -> dict:
        # TODO: Implement deterministic get method
        raise NotImplementedError(f"Search is not implemented for {self.__class__.__name__}")

    def search(self, name: str, results: int = 5) -> dict:
        raise NotImplementedError(f"Search is not implemented for {self.__class__.__name__}")

    def random(self) -> dict:
        return self._random_date()
