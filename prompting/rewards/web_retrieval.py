import re
import time
import numpy as np
from loguru import logger

from scipy import spatial
import trafilatura

from prompting.datasets.random_website import DDGDatasetEntry
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from prompting.base.dendrite import DendriteResponseEvent


_SEARCH_TERM_THRESH = 0.1
_VALID_URL_SCORE = 0.95


class WebRetrievalRewardModel(BaseRewardModel):
    def _cosine_similarity(self, reference: str, response: str) -> float:
        """Calculate the cosine similarity between sentence embeddings of the reference and completions."""
        reference_emb_flatten = self.embedding_model.encode(reference, to_numpy=True).flatten()
        response_emb_flatten = self.embedding_model.encode(response, to_numpy=True).flatten()
        return 1.0 - float(spatial.distance.cosine(reference_emb_flatten, response_emb_flatten))

    def reward(self, reference: DDGDatasetEntry, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Gives an exact reward of 1 if the response matches the reference, 0 otherwise"""
        completion: str = "\n".join(response_event.completions)
        timer_start = time.perf_counter()

        # Initial search term.
        search_term = reference.search_term
        # Actual text from the website.
        reference_content = reference.website_content
        # URL and the content provided in the completion.
        response_url, response_content = self._parse_response(completion)
        # Content scraped from the URL provided in the completion.
        response_url_scrape = self._extract_website_content(response_url)

        # Similarity between search term and reference content.
        search_reference_sim = self._cosine_similarity(reference=search_term, response=reference_content)
        # Similarity between search term and miner's scraped content.
        search_response_sim = self._cosine_similarity(reference=search_term, response=response_content)
        # If the URL provided in the completion is valid.
        valid_url_score = self._cosine_similarity(reference=response_content, response=response_url_scrape)

        response_reference_ratio = search_response_sim / search_reference_sim
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
        else:
            score = search_response_sim * valid_url_score

        return BatchRewardOutput(rewards=np.asarray([score]), timings=np.asarray([time.perf_counter() - timer_start]))

    def _extract_website_content(self, url) -> str:
        website = trafilatura.fetch_url(url)
        return trafilatura.extract(website)

    def _parse_response(self, completion: str) -> tuple[str, str]:
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
