import re
import time

import numpy as np
import trafilatura
from loguru import logger
from scipy import spatial

from prompting.base.dendrite import DendriteResponseEvent
from prompting.datasets.random_website import DDGDatasetEntry
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BatchRewardOutput

_SEARCH_TERM_THRESH = 0.3
_VALID_URL_SCORE = 0.8


class WebRetrievalRewardModel(RelevanceRewardModel):
    def _cosine_similarity(self, content1: str, content2: str) -> float:
        """Calculate the cosine similarity between sentence embeddings of the reference and completions."""
        reference_emb_flatten = self.embedding_model.encode(content1, to_numpy=True).flatten()
        response_emb_flatten = self.embedding_model.encode(content2, to_numpy=True).flatten()
        return 1.0 - float(spatial.distance.cosine(reference_emb_flatten, response_emb_flatten))

    # TODO: Change base class reference type to Reference pydantic model, in order to store additional data.
    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Score response website content and URL based on the similarity to the search term and reference content."""
        timer_start = time.perf_counter()
        completion: str = "\n".join(response_event.completions)

        if not completion:
            return BatchRewardOutput(rewards=np.asarray([0]), timings=np.asarray([time.perf_counter() - timer_start]))

        dataset_entry = DDGDatasetEntry.model_validate_json(reference)
        search_term = dataset_entry.search_term
        reference_content = dataset_entry.website_content

        # URL and the content provided in the completion.
        response_url, response_content = self._parse_response(completion)
        # Content scraped from the URL provided in the completion.
        response_url_scraped = self._extract_website_content(response_url)

        # Similarity between search term and miner's scraped content.
        search_response_sim = self._cosine_similarity(content1=search_term, content2=response_content)

        # If the URL provided in the completion is valid.
        valid_url_score = 0
        if response_url_scraped is not None:
            valid_url_score = self._cosine_similarity(content1=response_content, content2=response_url_scraped)

        # Similarity between search term and reference content.
        search_reference_sim = self._cosine_similarity(content1=search_term, content2=reference_content)
        response_reference_ratio = search_response_sim / search_reference_sim
        score = (search_response_sim + valid_url_score) / 2
        if abs(response_reference_ratio - 1) > _SEARCH_TERM_THRESH:
            logger.info(
                f"Reponse and reference scraped content relevance to the search term exceeds the threshold. "
                f"Similarity: response = {search_response_sim:.2f}; reference = {search_reference_sim:.2f}"
            )
            score = 0
        elif valid_url_score < _VALID_URL_SCORE:
            logger.info(
                f"Search term is not relevant to the scraped content, similarity {valid_url_score} < {_VALID_URL_SCORE}"
            )
            score = 0

        return BatchRewardOutput(rewards=np.asarray([score]), timings=np.asarray([time.perf_counter() - timer_start]))

    @staticmethod
    def _extract_website_content(url) -> str:
        website = trafilatura.fetch_url(url)
        return trafilatura.extract(website)

    @staticmethod
    def _parse_response(completion: str) -> tuple[str, str]:
        """Parse the completion text and extracts the URL and content.

        Args:
            completion: The text to parse.

        Returns:
            tuple: A tuple containing the URL (str or None) and the content (str).
        """
        url = None
        lines = completion.strip().split("\n")

        # First, try to find URL by parsing the last line that starts with "URL:".
        if lines and lines[-1].startswith("URL:"):
            url_line = lines.pop()
            url = url_line[len("URL:"):].strip()
        else:
            # Search for any line that starts with "URL:".
            url_found = False
            for idx, line in enumerate(lines):
                if line.startswith("URL:"):
                    url = line[len("URL:"):].strip()
                    # Remove the line containing the URL.
                    lines.pop(idx)
                    url_found = True
                    break
            # If still not found, search for any URL in the text.
            if not url_found:
                url_pattern = re.compile(r"https?://[^\s]+", re.IGNORECASE)
                match = url_pattern.search(completion)
                if match:
                    url = match.group(0)

        # Reconstruct the content without the URL line(s).
        content = "\n".join(lines).strip()
        return url, content
