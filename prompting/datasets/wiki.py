import random
import re
import sys
from functools import lru_cache
from queue import Full, Queue
from typing import ClassVar

import requests
import wikipedia
from bs4 import BeautifulSoup
from loguru import logger

from shared.base import BaseDataset, Context

# Create a queue called CACHED_ARTICLES to store wikipedia articles that have been fetched
CACHED_ARTICLES: Queue[Context] = Queue(maxsize=300)


# speed up page loading
@lru_cache(maxsize=1000)
def _get_page(
    title: str, pageid: str | None = None, auto_suggest: bool = False, redirect: bool = True, seed: int | None = None
) -> wikipedia.WikipediaPage:
    """Cached Wikipedia page loading."""
    try:
        page = wikipedia.page(title=title, pageid=pageid, auto_suggest=auto_suggest, redirect=redirect)
        return page

    except wikipedia.DisambiguationError as e:
        logger.debug(f"{e.__class__.__name__} loading page {title!r}: {e}")
        # exc info contains a tuple of (requested_title: str, possible_matches: list[str])
        pages = sys.exc_info()[1].args[1]
        if not isinstance(pages, list):
            return None
        title = random.Random(seed).choice(pages)
        return _get_page(title, auto_suggest=auto_suggest, redirect=redirect)

    except wikipedia.PageError as e:
        logger.warning(f"{e.__class__.__name__} loading page {title!r}: {e}")
        if not auto_suggest:
            return _get_page(title, auto_suggest=True, redirect=redirect)
        return None


def _get_random_titles(pages: int = 10) -> list:
    return wikipedia.random(pages=pages)


@lru_cache(maxsize=1000)
def _wikipedia_search(name: str, results: wikipedia.WikipediaPage) -> list:
    """Cached Wikipedia search."""
    return wikipedia.search(name, results=results)


def get_article_sections(title: str) -> dict[str, str]:
    # Fetch the HTML content of the Wikipedia article
    url = f"https://en.wikipedia.org/wiki/{title}"
    response = requests.get(url)
    html_content = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    sections = {}
    for section in soup.find_all("h2"):
        if (p_tag := section.find_next("p")) is not None:
            sections[section.text] = p_tag.text

    return sections


def process_page(
    page: wikipedia.WikipediaPage, exclude_sections: list | None = None, valid_section: callable = None
) -> dict:
    """Process a Wikipedia page and return a dictionary of sections with their content.

    Args:
        page: wikipedia.WikipediaPage
        valid_header: callable to determine if a section header is valid
        valid_content: callable to determine if a section content is valid
    Returns:
        dict: dictionary of sections and their content. Note that keys are tuples (header, section_title)
    """
    title = page.title

    sections = get_article_sections(title)

    # Filter out the section keys that are in the exclude list
    if exclude_sections:
        sections = {k: v for k, v in sections.items() if k not in exclude_sections}

    valid_sections = [
        (key, value) for key, value in sections.items() if not valid_section or valid_section(sections[key])
    ]

    if valid_sections:
        return random.choice(valid_sections), sections.keys()
    else:
        return None, sections.keys()


def most_relevant_links(
    page: wikipedia.WikipediaPage, num_links: int = 10, num_summary_words: int = 50, return_scores: bool = False
) -> list:
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


def filter_categories(categories: list[str], exclude: list[str] = [], include: list[str] = []):
    """Filter categories based on a list of categories to exclude and/or include."""
    if exclude:
        categories = [cat for cat in categories if not re.search("|".join(exclude), cat, re.IGNORECASE)]
    if include:
        categories = [cat for cat in categories if re.search("|".join(include), cat, re.IGNORECASE)]
    return categories


class WikiDataset(BaseDataset):
    """Wikipedia dataset. Uses the wikipedia python api to fetch articles and sections."""

    EXCLUDE_HEADERS = ("See also", "References", "Further reading", "External links")
    EXCLUDE_CATEGORIES = ("articles", "wiki", "pages", "cs1")
    name: ClassVar[str] = "wikipedia"
    EXCLUDE_HEADERS: tuple = ("See also", "References", "Further reading", "External links")
    EXCLUDE_CATEGORIES: tuple = ("articles", "wikipedia", "pages", "cs1")
    min_length_words: int = 20
    max_links: int = 10

    def get(
        self,
        name: str,
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
        selected_section, _ = process_page(
            page,
            exclude_sections=exclude,
            valid_section=lambda x: len(x.split()) >= self.min_length_words,
        )
        if not selected_section:
            return None
        header, section_title = selected_section

        section_length = len(selected_section[1].split())

        context = Context(
            title=name,
            topic=header or section_title,
            subtopic=section_title,
            content=section_title,
            internal_links=list(filter(lambda x: x not in exclude, page.sections)),
            external_links=most_relevant_links(page, num_links=self.max_links),
            tags=filter_categories(page.categories, exclude=self.EXCLUDE_CATEGORIES),
            source=name,
            extra={
                "url": page.url,
                "page_length": len(page.content.split()),
                "section_length": section_length,
            },
        )
        try:
            CACHED_ARTICLES.put(context, block=False)
        except Full:
            logger.debug("Cache is full. Skipping article until cache is emptied.")
        return context

    def search(self, name, results=3) -> Context:
        titles = _wikipedia_search(name, results=results)
        title = random.choice(titles)
        return self.get(title)

    def random(self, pages=10) -> dict:
        titles = _get_random_titles(pages=pages)
        for title in titles[: self.max_tries]:
            if context := self.get(title):
                return context
        return None
