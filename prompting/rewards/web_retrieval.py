"""Expected miner's response is a JSON object with the following keys: url, content, relevant.

Example response:
{
    "url": "https://www.example.com",
    "content": "This is the content of the website. This is the section we are interested in.",
    "relevant": "This is the section we are interested in.",
}
"""
import json
import time

import numpy as np
from loguru import logger
from scipy import spatial

from prompting.base.dendrite import DendriteResponseEvent
from prompting.datasets.random_website import DDGDataset, DDGDatasetEntry
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BatchRewardOutput

_SEARCH_TERM_THRESH = 0.4
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

        # URL and the content provided in the completion.
        response_url, response_content, response_relevant = self._parse_response(completion)
        if response_url is None or response_content is None:
            return BatchRewardOutput(rewards=np.asarray([0]), timings=np.asarray([time.perf_counter() - timer_start]))

        # Content scraped from the URL provided in the completion.
        response_url_scraped = DDGDataset.extract_website_content(response_url)
        if not response_url_scraped or len(response_url_scraped) == 0:
            logger.debug(f"Failed to extract miner's content from website: {response_url}")
            return BatchRewardOutput(rewards=np.asarray([0]), timings=np.asarray([time.perf_counter() - timer_start]))

        dataset_entry = DDGDatasetEntry.model_validate_json(reference)
        search_term = dataset_entry.search_term
        reference_content = dataset_entry.website_content

        # Similarity between search term and miner's scraped content.
        search_response_sim = self._cosine_similarity(content1=search_term, content2=response_content)

        # Similarity between search term and relevant section of content.
        search_relevant_sim = 0
        if response_relevant is not None:
            search_relevant_sim = self._cosine_similarity(content1=search_term, content2=response_relevant)

        # If the URL provided in the completion is valid.
        valid_url_score = 0
        if response_url_scraped is not None:
            valid_url_score = self._cosine_similarity(content1=response_content, content2=response_url_scraped)

        # Similarity between search term and reference content.
        search_reference_sim = self._cosine_similarity(content1=search_term, content2=reference_content)
        score = (search_response_sim + valid_url_score + search_relevant_sim) / 3
        if abs(search_response_sim - search_reference_sim) > _SEARCH_TERM_THRESH:
            logger.info(
                f"Response and reference scraped content relevance to the search term exceeds the threshold. "
                f"Similarity: response = {search_response_sim:.2f}; reference = {search_reference_sim:.2f}"
            )
            score = 0
        elif valid_url_score < _VALID_URL_SCORE:
            # If provided URL does not contain content.
            logger.info(
                f"Search term is not relevant to the scraped content, similarity {valid_url_score} < {_VALID_URL_SCORE}"
            )
            score = 0
        elif response_relevant is not None and len(response_relevant) > len(response_content):
            logger.info(
                "Relevant section is longer than the whole website content "
                f"{len(response_relevant)} > {len(response_content)}"
            )
            score = 0

        return BatchRewardOutput(rewards=np.asarray([score]), timings=np.asarray([time.perf_counter() - timer_start]))

    @staticmethod
    def _parse_response(completion: str) -> tuple[str | None, ...]:
        try:
            data = json.loads(completion)
            response_url = data.get("url")
            response_content = data.get("content")
            response_relevant = data.get("relevant")
        except json.JSONDecodeError:
            response_url = None
            response_content = None
            response_relevant = None
        return response_url, response_content, response_relevant
