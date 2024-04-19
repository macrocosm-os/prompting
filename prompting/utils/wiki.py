import re
import sys
import random
import bittensor as bt
import wikipedia as wiki
from typing import Dict, List
from functools import lru_cache


# speed up page loading
@lru_cache(maxsize=1000)
def _get_page(
    title, pageid=None, auto_suggest=False, redirect=True, seed=None
) -> wiki.WikipediaPage:
    """Cached Wikipedia page loading."""
    try:
        page = wiki.page(
            title=title, pageid=pageid, auto_suggest=auto_suggest, redirect=redirect
        )
        # create sections manually if not found
        if not page.sections:
            page._sections = [
                line.strip("= ")
                for line in page.content.splitlines()
                if re.search(r"=+\s+.*\s+=+", line)
            ]
        return page

    except wiki.DisambiguationError as e:
        bt.logging.debug(f"{e.__class__.__name__} loading page {title!r}: {e}")
        # exc info contains a tuple of (requested_title: str, possible_matches: List[str])
        pages = sys.exc_info()[1].args[1]
        if not type(pages) == list:
            return None
        title = random.Random(seed).choice(pages)
        return _get_page(title, auto_suggest=auto_suggest, redirect=redirect)

    except wiki.PageError as e:
        bt.logging.warning(f"{e.__class__.__name__} loading page {title!r}: {e}")
        if not auto_suggest:
            return _get_page(title, auto_suggest=True, redirect=redirect)
        return None


@lru_cache(maxsize=1000)
def _get_random_titles(pages=10, seed=42) -> List:
    """Cached wikipedia random page. Approximately deterministic random titles. This is useful for testing.
    NOTE: the actually cached result will change each session, but the result will be the same within a session.
    """
    return wiki.random(pages=pages)


@lru_cache(maxsize=1000)
def _wiki_search(name, results) -> List:
    """Cached Wikipedia search."""
    return wiki.search(name, results=results)


def process_page(
    page, valid_header: callable = None, valid_content: callable = None
) -> Dict:
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
        if (valid_header and not valid_header(header)) or (
            valid_content and not valid_content(content)
        ):
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
        iou = len(summary_words.intersection(link_words)) / len(
            summary_words.union(link_words)
        )
        link_scores[link] = iou / len(link.split())

    sorted_links = sorted(link_scores.items(), key=lambda x: x[1], reverse=True)
    if return_scores:
        return sorted_links[:num_links]

    return [link for link, _ in sorted_links[:num_links]]


def filter_categories(categories, exclude=None, include=None):
    """Filter categories based on a list of categories to exclude and/or include."""
    if exclude:
        categories = [
            cat
            for cat in categories
            if not re.search("|".join(exclude), cat, re.IGNORECASE)
        ]
    if include:
        categories = [
            cat
            for cat in categories
            if re.search("|".join(include), cat, re.IGNORECASE)
        ]
    return categories
