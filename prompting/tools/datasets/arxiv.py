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
import sys
import fitz
import random
import datetime
from dataclasses import dataclass
import bittensor as bt
import arxiv
from typing import Dict, Union, List, Tuple
from queue import Queue, Full, Empty
from functools import lru_cache
from .base import Dataset
from ..selector import Selector


# Create a queue called CACHED_ARTICLES to store arxiv articles that have been fetched
CACHED_ARTICLES = Queue(maxsize=300)

client = arxiv.Client()


@dataclass
class ArxivPaper:

    # Define common section headers to search for
    section_headers = [
        r"(?i)\babstract\b",
        r"(?i)\bintroduction\b",
        r"(?i)\bmethods\b",
        r"(?i)\bmethodology\b",
        r"(?i)\bresults\b",
        r"(?i)\bdiscussion\b",
        r"(?i)\bconclusion\b",
        r"(?i)\breferences\b",
    ]

    def __init__(self, paper, text):
        self.paper = paper
        self.text = text

        self.parse(text)

    def parse(self, text):
        # Find sections using regular expressions
        sections = {}
        for header in self.section_headers:
            match = re.search(header, text)
            if match:
                sections[match.group().lower()] = match.start()

        # Sort sections by their position in the text
        sorted_sections = sorted(sections.items(), key=lambda x: x[1])

        # Extract sections
        paper_sections = {}
        for i in range(len(sorted_sections)):
            section_name = sorted_sections[i][0]
            section_start = sorted_sections[i][1]
            section_end = (
                sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
            )
            paper_sections[section_name] = text[section_start:section_end].strip(
                f" \n{section_name}"
            )

        self.sections = paper_sections


# speed up page loading
@lru_cache(maxsize=1000)
def _get_text(paper, seed=None) -> arxiv.Result:
    """Cached Arxiv paper loading."""
    try:
        pdf_filename = paper.download_pdf()
        # Read the PDF file
        with fitz.open(pdf_filename) as pdf_document:
            num_pages = pdf_document.page_count
            text = ""
            for page_num in range(num_pages):
                page = pdf_document.load_page(page_num)
                text += page.get_text()

        # If paper is terribly formatted, return None
        if text.count("\n") / len(text.split()) > 0.3:
            bt.logging.warning(f"Paper {paper!r} is poorly formatted. Rejecting.")
            return None

        # Returns the pdf text
        return text

    except Exception as e:
        bt.logging.warning(f"{e.__class__.__name__} loading paper {paper!r}: {e}")
        return None


class ArxivDataset(Dataset):
    """Arxiv dataset. Uses the Arxiv python api to fetch papers and sections."""

    name = "arxiv"
    INCLUDE_HEADERS = (
        "Abstract",
        "Introduction",
        "Background",
        "Method",
        "Methodology",
        "Results",
        "Discussion",
        "Conclusion",
    )
    EXCLUDE_HEADERS = ("See also", "References", "Appendix")

    def __init__(
        self,
        min_length_words: int = 50,
    ):
        """
        Args:
            min_length_words (int, optional): Minimum section length. Defaults to 50.
        """
        self.min_length_words = min_length_words

    def get(
        self,
        name: str,
        selector: Selector = None,
        include: List = None,
        exclude: List = None,
        **kwargs,
    ) -> Dict:
        """Get a specified Arxiv page and extract a section based on the selector.

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

        text = _get_text(name=name, **kwargs)
        if text is None:
            return None

        exclude = (exclude or []) + list(self.EXCLUDE_HEADERS)
        if not include:
            include = self.INCLUDE_HEADERS

        paper = ArxivPaper(
            paper,
            text,
            # Only return the specified sections
            valid_header=lambda x: x not in exclude and (not include or x in include),
            # Only return sections with a minimum number of words
            valid_content=lambda x: len(x.split()) >= self.min_length_words,
        )
        sections = paper.sections

        if not sections:
            return None

        key = header, section_title = selector(list(sections.keys()))
        content = "\n".join(sections[key])
        section_length = len(content.split())
        context = {
            "title": name,  # title of arxiv paper
            "topic": header or section_title,  # title of section
            "subtopic": section_title,
            "content": content,
            "internal_links": list(filter(lambda x: x not in exclude, paper.sections)),
            "external_links": [],
            "tags": [],
            "source": "Wikipedia",
            "extra": {
                "url": paper.url,
                "paper_length": len(text.split()),
                "section_length": section_length,
            },
        }
        try:
            CACHED_ARTICLES.put(context, block=False)
        except Full:
            bt.logging.debug("Cache is full. Skipping article until cache is emptied.")
        return context

    def search(self, name, results=3, selector: Selector = None) -> Dict:

        search = arxiv.Search(
            query=name, max_results=results, sort_by=arxiv.SortCriterion.Relevance
        )

        results = list(client.results(search))
        random_paper = selector(results)
        return random_paper.title

    def random(self, pages=10, seed=None, selector: Selector = None, **kwargs) -> Dict:

        search = arxiv.Search(
            query="all:random", max_results=pages, sort_by=arxiv.SortCriterion.Relevance
        )

        results = list(client.results(search))
        random_paper = selector(results)

        return self.get(random_paper, selector=selector)
