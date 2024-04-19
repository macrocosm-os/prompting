from prompting.utils import async_wiki_utils as async_wiki_utils
from typing import List
from .base import BatchDataset
from prompting.tools.datasets import Context
from prompting.utils.async_wiki_utils import get_batch_random_sections


class BatchWikiDataset(BatchDataset):
    """Wikipedia dataset. Uses the wikipedia python api to fetch articles and sections."""

    def __init__(
        self,
        batch_size: int = 16,
    ):
        """
        Args:
            min_length_words (int, optional): Minimum section length. Defaults to 50.
            max_links (int, optional): _description_. Defaults to 10.
        """
        self.batch_size = batch_size

    async def random_batch(self) -> List[Context]:
        contexts = await get_batch_random_sections(self.batch_size)
        return contexts
