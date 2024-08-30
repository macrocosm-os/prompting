import random
from typing import Optional
from duckduckgo_search import DDGS
import trafilatura
from prompting.datasets.base import BaseDataset, DatasetEntry
from loguru import logger


def load_words(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        return file.read().splitlines()


MAX_CHARS = 5000
WORDS = load_words("prompting/datasets/english_words.txt")


class DDGDatasetEntry(DatasetEntry):
    search_term: str
    website_url: str
    website_content: str


class DDGDataset(BaseDataset):
    @staticmethod
    def search_random_term(retries: int = 3) -> tuple[Optional[str], Optional[list[dict[str, str]]]]:
        try:
            ddg = DDGS()
            for _ in range(retries):
                random_words = " ".join(random.sample(WORDS, 5))
                results = list(ddg.text(random_words))
                if results:
                    return random_words, results
        except Exception as ex:
            logger.error(f"Failed to get search results from DuckDuckGo: {ex}")
        return None, None

    @staticmethod
    def extract_website_content(url: str) -> Optional[str]:
        try:
            website = trafilatura.fetch_url(url)
            extracted = trafilatura.extract(website)
            return extracted[:MAX_CHARS] if extracted else None
        except Exception as ex:
            logger.error(f"Failed to extract content from website {url}: {ex}")

    def next(self) -> Optional[DDGDatasetEntry]:
        search_term, results = self.search_random_term(retries=3)
        if not results:
            return None
        website_url = random.choice(results)["href"]
        website_content = self.extract_website_content(website_url)
        if not website_content or len(website_content) == 0:
            logger.error(f"Failed to extract content from website {website_url}")
            return None

        return DDGDatasetEntry(search_term=search_term, website_url=website_url, website_content=website_content)

    def get(self) -> Optional[DDGDatasetEntry]:
        return self.next()

    def random(self) -> DatasetEntry:
        return self.next()
