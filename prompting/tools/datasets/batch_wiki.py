import re
import sys
import random
import datetime
import bittensor as bt
import wikipedia as wiki
from prompting.utils import async_wiki as async_wiki_utils
from typing import Dict, Union, List, Tuple
from functools import lru_cache
from .base import Dataset, BatchDataset
from ..selector import Selector
from prompting.tools.datasets import Context




@lru_cache(maxsize=1000)
def _get_random_titles(pages=10, seed=42) -> List:
    """Cached wikipedia random page. Approximately deterministic random titles. This is useful for testing.
    NOTE: the actually cached result will change each session, but the result will be the same within a session.
    """
    return wiki.random(pages=pages)




class BatchWikiDataset(BatchDataset):
    """Wikipedia dataset. Uses the wikipedia python api to fetch articles and sections."""

    EXCLUDE_HEADERS = ("See also", "References", "Further reading", "External links")
    EXCLUDE_CATEGORIES = ("articles", "wiki", "pages", "cs1")

    def __init__(
        self,
        batch_size: int = 16,
        min_length_words: int = 50,
        max_links: int = 10,
    ):
        """
        Args:
            min_length_words (int, optional): Minimum section length. Defaults to 50.
            max_links (int, optional): _description_. Defaults to 10.
        """
        self.batch_size = batch_size
        self.min_length_words = min_length_words
        self.max_links = max_links
     
    async def get_multiple_pages(
        self,
        titles: List[str],
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
        pages = await async_wiki_utils.fetch_pages(titles, **kwargs)
        
        # Only return a sections with a minimum number of words
        exclude = (exclude or []) + list(self.EXCLUDE_HEADERS)
        
        # TODO: FIX THE RETURN FOR PROCESS PAGES TO BE A MANAGABLE TYPE
        sections = await async_wiki_utils.process_pages(
            pages,
            valid_header=lambda x: x not in exclude and (not include or x in include),
            valid_content=lambda x: len(x.split()) >= self.min_length_words,
        )
        

        key = header, section_title = selector(list(sections.keys()))
        content = "\n".join(sections[key])
        section_length = len(content.split())
        return {
            "title": name,  # title of wiki article
            "topic": header or section_title,  # title of wiki section
            "subtopic": section_title,
            "content": content,
            "internal_links": list(filter(lambda x: x not in exclude, page.sections)),
            "external_links": most_relevant_links(page, num_links=self.max_links),
            "tags": filter_categories(page.categories, exclude=self.EXCLUDE_CATEGORIES),
            "source": "Wikipedia",
            "extra": {
                "url": page.url,
                "page_length": len(page.content.split()),
                "section_length": section_length,
            },
        }

    async def random_batch(self, seed=None, selector: Selector = None, **kwargs) -> List[Context]:
        """Get random batch of wikipedia pages."""
        random_titles = (
            wiki.random(pages=self.batch_size)
            if seed is None
            else _get_random_titles(pages=self.batch_size, seed=seed)
        )        
        
        return await self.get_multiple_pages(random_titles, selector=selector)