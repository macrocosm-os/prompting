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
import wikipedia as wiki
from typing import Dict, List
from prompting.utils import wiki as wiki_utils
from .base import Dataset
from ..selector import Selector


class WikiDataset(Dataset):
    """Wikipedia dataset. Uses the wikipedia python api to fetch articles and sections."""

    name = "wiki"
    EXCLUDE_HEADERS = ("See also", "References", "Further reading", "External links")
    EXCLUDE_CATEGORIES = ("articles", "wiki", "pages", "cs1")

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

    def get(
        self,
        name: str,
        selector: Selector = None,
        include: List = None,
        exclude: List = None,
        **kwargs,
    ) -> Dict:
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

        page = wiki_utils._get_page(title=name, **kwargs)
        if page is None:
            return None

        # Only return a sections with a minimum number of words
        exclude = (exclude or []) + list(self.EXCLUDE_HEADERS)
        sections = wiki_utils.process_page(
            page,
            valid_header=lambda x: x not in exclude and (not include or x in include),
            valid_content=lambda x: len(x.split()) >= self.min_length_words,
        )
        if not sections:
            return None

        key = header, section_title = selector(list(sections.keys()))
        content = "\n".join(sections[key])
        section_length = len(content.split())
        return {
            "title": name,  # title of wiki article
            "topic": header or section_title,  # title of wiki section
            "subtopic": section_title,
            "content": content,
            "internal_links": list(filter(lambda x: x not in exclude, page.sections)),
            "external_links": wiki_utils.most_relevant_links(
                page, num_links=self.max_links
            ),
            "tags": wiki_utils.filter_categories(
                page.categories, exclude=self.EXCLUDE_CATEGORIES
            ),
            "source": "Wikipedia",
            "extra": {
                "url": page.url,
                "page_length": len(page.content.split()),
                "section_length": section_length,
            },
        }

    def search(self, name, results=3, selector: Selector = None) -> Dict:
        titles = wiki_utils._wiki_search(name, results=results)
        title = selector(titles)
        return self.get(title, selector=selector)

    def random(self, pages=10, seed=None, selector: Selector = None, **kwargs) -> Dict:
        titles = (
            wiki.random(pages=pages)
            if seed is None
            else wiki_utils._get_random_titles(pages=pages, seed=seed)
        )
        title = selector(titles)
        return self.get(title, selector=selector)


class WikiDateDataset(Dataset):
    name = "wiki_date"
    INCLUDE_HEADERS = ("Events", "Births", "Deaths")
    MONTHS = (
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
    EXCLUDE_CATEGORIES = ("articles", "wiki", "pages", "cs1")

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
        return random_date.strftime("%B %-d")  # E.g., "January 1"

    def get(
        self,
        name,
        pageid=None,
        auto_suggest=False,
        redirect=False,
        selector: Selector = None,
    ) -> Dict:
        # Check that name is correctly formatted e.g., "January 1"
        date = name.split(" ")
        assert (
            len(date) == 2
        ), f"Date should be in the format 'Month D[D]' (e.g., 'January 1' or 'March 28'), but got {name!r}"
        assert (
            date[0] in self.MONTHS
        ), f"Month should be one of {self.MONTHS}, but got {date[0]!r}"
        assert date[1].isdigit(), f"Day should be a number, but got {date[1]!r}"

        page = wiki_utils._get_page(
            title=name, pageid=pageid, auto_suggest=auto_suggest, redirect=redirect
        )
        if page is None:
            return None

        # Only return a sections which contain event-like format
        # e.g. "1999 - Some event happened"
        sections = wiki_utils.process_page(
            page,
            valid_header=lambda x: x in self.INCLUDE_HEADERS,
            valid_content=lambda x: any(
                [re.search(r"^\d+", line) for line in x.splitlines()]
            ),
        )
        if not sections:
            return None

        key = header, section_title = selector(list(sections.keys()))
        line = selector(sections[key])
        year, *event = line.replace("\u2013", "-").split("-")
        links = [link for link in page.links if link in line]

        return {
            "title": name,  # title of wiki article
            "topic": header or section_title,  # title of wiki section
            "subtopic": year.strip(),
            "content": "-".join(event).strip(". "),
            "internal_links": list(sections.keys()),
            "external_links": links,
            "tags": wiki_utils.filter_categories(
                page.categories, exclude=WikiDataset.EXCLUDE_CATEGORIES
            ),
            "source": "Wikipedia",
            "extra": {
                "url": page.url,
                "year": year,
                "event": event,
                "line": line,
                "date": date + [year],
                "section_title": section_title,
            },
        }

    def search(self, name, results=5, selector: Selector = None) -> Dict:
        raise NotImplementedError(
            f"Search is not implemented for {self.__class__.__name__}"
        )

    def random(self, selector: Selector = None, **kwargs) -> Dict:
        date = self._random_date()
        return self.get(date, selector=selector)
