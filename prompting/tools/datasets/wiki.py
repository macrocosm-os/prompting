# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re
import random
import datetime
import bittensor as bt
import wikipedia as wiki
from typing import Dict, Union, List, Tuple

from functools import lru_cache
from .base import Dataset
from ..selector import Selector

# TODO: we need a way to do an 'internal' step through a current document (e.g another section or line of a current context)
# Since we are caching the page fetch we should just call get again with the same page name and hope that it selects a different section/line
# this means we use the context.title field to search again, rather than internal links... but we need to make sure that the context is different
# DO WE NEED INTERAL LINKS??

# speed up page loading
@lru_cache(maxsize=1000)
def _get_page(title, pageid=None, auto_suggest=True, redirect=True) -> wiki.WikipediaPage:
    try:
        return wiki.page(title=title, pageid=pageid, auto_suggest=auto_suggest, redirect=redirect)
    except (wiki.PageError, wiki.DisambiguationError) as e:
        bt.logging.error(f"{e.__class__.__name__} loading page {title!r}: {e}")
        return None

def process_page(page, valid_header: callable = None, valid_content: callable = None) -> Dict:
    """Process a Wikipedia page and return a dictionary of sections with their content.

    Args:
        page: wikipedia.WikipediaPage
        valid_header: callable to determine if a section header is valid
        valid_content: callable to determine if a section content is valid
    Returns:
        dict: dictionary of sections and their content. Note that keys are tuples (header, section_title)
    """
    header = ''
    sections = {}
    for section_title in page.sections:
        content = page.section(section_title)
        if not content:
            header = section_title
            continue

        # Filter out sections that don't match the headers and/or are not valid
        if (valid_header and not valid_header(header)) or \
            (valid_content and not valid_content(content)):
            continue

        key = (header, section_title)
        sections[key] = content.splitlines()

    if not sections:
        raise Exception(f"No valid sections found in page {page.title!r} ({page.url})")

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


class WikiDataset(Dataset):
    """Wikipedia dataset. Uses the wikipedia python api to fetch articles and sections."""

    EXCLUDE_HEADERS = ('See also', 'References', 'Further reading', 'External links')

    def __init__(
        self,
        min_length_words: int = 50,
        max_links: int = 10,
    ):
        """
        Args:
            min_length_words (int, optional): Minimum section length. Defaults to 50.
            max_links (int, optional): _description_. Defaults to 10.
        """
        self.min_length_words = min_length_words
        self.max_links = max_links


    def get(self, name, pageid=None, auto_suggest=True, redirect=True, selector: Selector = None, include: List = None, exclude: List = None) -> Dict:
        """Get a specified Wikipedia page and extract a section based on the selector.

        Args:
            name (_type_): _description_
            pageid (_type_, optional): _description_. Defaults to None.
            auto_suggest (bool, optional): _description_. Defaults to True.
            redirect (bool, optional): _description_. Defaults to True.
            selector (Selector, optional): _description_. Defaults to None.
            include (List, optional): _description_. Defaults to None.
            exclude (List, optional): _description_. Defaults to None.

        Returns:
            Dict: _description_
        """

        page = _get_page(title=name, pageid=pageid, auto_suggest=auto_suggest, redirect=redirect)
        if page is None:
            return None

        # Only return a sections with a minimum number of words
        exclude = (exclude or []) + list(self.EXCLUDE_HEADERS)
        sections = process_page(page,
                                valid_header=lambda x: x not in exclude and (not include or x in include),
                                valid_content=lambda x: len(x.split())>=self.min_length_words
                                )

        key = header, section_title = selector(list(sections.keys()))
        content = '\n'.join(sections[key])
        section_length = len(content.split())
        return {
            "title": name, # title of wiki article
            "topic": header or section_title, # title of wiki section
            'subtopic': section_title,
            'content': content,
            'internal_links': list(filter(lambda x: x not in exclude, page.sections)),
            'external_links': most_relevant_links(page, num_links=self.max_links), #TODO: only take links from selected section(?)
            'source': 'Wikipedia',
            'extra': {'url': page.url, 'page_length': len(page.content.split()), 'section_length': section_length},
        }

    def search(self, name, results=3, selector: Selector = None) -> Dict:
        titles = wiki.search(name, results=results)
        title = selector(titles)
        return self.get(title, selector=selector)

    def random(self, pages=10, selector: Selector = None) -> Dict:
        titles = wiki.random(pages=pages)
        title = selector(titles)
        return self.get(title, selector=selector)




class WikiDateDataset(Dataset):

    INCLUDE_HEADERS = ("Events", "Births", "Deaths")
    MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")

    def __init__(self, max_tries: int = 10, seed=None):
        self.max_tries = max_tries
        self.seed = seed
        self.rng = random.Random(seed)

    def _random_date(self, year: int = None, month: int = None) -> int:
        """Returns a random date in the format "Month_DD" (e.g., "January_01")."""
        if year is None:
            year = self.rng.randint(0, 2024)
        if month is None:
            month = self.rng.randint(1, 12)

        max_days = 31 if month in (1, 3, 5, 7, 8, 10, 12) else 30
        max_days = max_days if month != 2 else 29

        day = self.rng.randint(1, max_days)

        random_date = datetime.date(year, month, day)
        # Step 2: Format the date for Wikipedia URL
        return random_date.strftime("%B_%d")  # E.g., "January_01"

    def get(self, name, pageid=None, auto_suggest=True, redirect=True, selector: Selector = None) -> Dict:

        # Check that name is correctly formatted e.g., "January_01"
        date = name.split('_')
        assert len(date)==2, f"Date should be in the format 'Month_DD' (e.g., 'January_01'), but got {name!r}"
        assert date[0] in self.MONTHS, f"Month should be one of {self.MONTHS}, but got {date[0]!r}"
        assert date[1].isdigit(), f"Day should be a number, but got {date[1]!r}"

        page = _get_page(title=name, pageid=pageid, auto_suggest=auto_suggest, redirect=redirect)
        if page is None:
            return None

        # Only return a sections which contain event-like format
        # e.g. "1999 - Some event happened"
        sections = process_page(page,
                                valid_header=lambda x: x in self.INCLUDE_HEADERS,
                                valid_content=lambda x: any([re.search(r'^\d+',line) for line in x.splitlines()])
                                )

        key = header, section_title = selector(list(sections.keys()))
        line = selector(sections[key])
        year, *event = line.replace(u'\u2013', '-').split('-')
        links = [link for link in page.links if link in line]

        return {
            "title": name, # title of wiki article
            "topic": header or section_title, # title of wiki section
            'subtopic': year.strip(),
            'content': '-'.join(event).strip(),
            'internal_links': list(sections.keys()), # unsure what to do for DQA->DQA. Same date events? Another random date? TBD
            'external_links': links,
            'source': 'Wikipedia',
            'extra': {'url': page.url, 'year': year, 'event': event, 'line': line, 'date': date+[year], 'section_title': section_title},
        }

    def search(self, name, results=5, selector: Selector = None) -> Dict:
        raise NotImplementedError("Search is not implemented for WikiDateDataset")

    def random(self, selector: Selector = None, **kwargs) -> Dict:
        date = self._random_date()
        return self.get(date, selector=selector)

