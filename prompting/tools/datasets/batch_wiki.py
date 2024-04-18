from prompting.utils import async_wiki_utils as async_wiki_utils
from typing import List
from .base import BatchDataset
from prompting.tools.datasets import Context
from prompting.utils.custom_async_wiki import get_batch_random_sections


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
     
  
    async def random_batch(self) -> List[Context]:        
        contexts = await get_batch_random_sections()
        return contexts
