"""Expected miner's response is a JSON object with the following keys: url, content, relevant.

Example response:
{
    "url": "https://www.example.com",
    "content": "This is the content of the website. This is the section we are interested in.",
    "relevant": "This is the section we are interested in.",
}
"""

import json
from pydantic import BaseModel

import numpy as np
from loguru import logger
from scipy import spatial
from thefuzz import fuzz

from prompting.datasets.random_website import DDGDataset, DDGDatasetEntry
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BatchRewardOutput
from prompting.tasks.base_task import BaseTextTask
from shared.dendrite import DendriteResponseEvent

MIN_RELEVANT_CHARS = 300
MIN_MATCH_THRESHOLD = 98


class WebsiteResult(BaseModel):
    url: str
    content: str
    relevant: str


class WebRetrievalRewardModel(RelevanceRewardModel):
    def _cosine_similarity(self, content1: str, content2: str) -> float:
        """Calculate the cosine similarity between sentence embeddings of the reference and completions."""
        reference_emb_flatten = self.embedding_model.encode(content1, to_numpy=True).flatten()
        response_emb_flatten = self.embedding_model.encode(content2, to_numpy=True).flatten()
        return 1.0 - float(spatial.distance.cosine(reference_emb_flatten, response_emb_flatten))

    def score_website_result(
        self, dataset_entry: DDGDatasetEntry, response_url: str, response_content: str, response_relevant: str
    ) -> float:
        if not response_url or not response_content or not response_relevant:
            return 0

        # Content scraped from the URL provided in the completion.
        reference_website_content = DDGDataset.extract_website_content(response_url)
        if not reference_website_content or len(reference_website_content) == 0:
            logger.debug(f"Failed to extract miner's content from website: {response_url}")
            return 0

        if fuzz.token_set_ratio(response_content, reference_website_content) < MIN_MATCH_THRESHOLD:
            logger.info("Miner returned text that doesn't match the website, scoring 0")
            return 0

        if len(response_relevant) > len(response_content) or len(response_relevant) < MIN_RELEVANT_CHARS:
            logger.info(
                f"Relevant section is too short (<{MIN_RELEVANT_CHARS} chars) or longer than the whole website content "
                f"{len(response_relevant)} > {len(response_content)}"
            )
            return 0

        # Similarity between search term and relevant section of content.
        if response_relevant is not None:
            score = self._cosine_similarity(content1=dataset_entry.query, content2=response_relevant)
            print(score)
        return score

    def score_miner_response(
        self, dataset_entry: DDGDatasetEntry, completion: str, task: BaseTextTask | None = None
    ) -> float:
        scores = []
        miner_websites: list[WebsiteResult] = self._parse_response(completion)
        unique_websites = np.unique([website.url for website in miner_websites])
        if unique_websites.size != len(miner_websites) and unique_websites.size != task.target_results:
            #logger.warning("Miner returned multiple websites with the same URL")
            return 0

        for website in miner_websites:
            scores.append(self.score_website_result(dataset_entry, website.url, website.content, website.relevant))

        if scores:
            return np.mean(scores)
        return 0

    # TODO: Change base class reference type to Reference pydantic model, in order to store additional data.
    def reward(
        self, reference: str, response_event: DendriteResponseEvent, task: BaseTextTask | None = None, **kwargs
    ) -> BatchRewardOutput:
        """Score response website content and URL based on the similarity to the search term and reference content."""
        rewards: list[float] = []
        timings: list[float] = []
        dataset_entry = DDGDatasetEntry.model_validate_json(json.loads(reference))
        if not dataset_entry.query:
            # if the dataset doesn't have a query, we can't score the completions
            return BatchRewardOutput(
                rewards=np.array([0] * len(response_event.completions)),
                timings=np.array([0] * len(response_event.completions)),
            )

        for completion in response_event.completions:
            rewards.append(self.score_miner_response(dataset_entry, completion, task=task))
            timings.append(0)

        print(rewards, timings, flush=True)
        return BatchRewardOutput(rewards=np.array(rewards), timings=np.array(timings))

    @staticmethod
    def _parse_response(completion: str) -> tuple[str | None, ...]:
        result = []
        try:
            data = json.loads(completion)
            if not isinstance(data, list) and isinstance(data, dict):
                data = [data]
            for website in data:
                response_url = website.get("url")
                response_content = website.get("content")
                response_relevant = website.get("relevant")
                result.append(WebsiteResult(url=response_url, content=response_content, relevant=response_relevant))
            return result
        except json.JSONDecodeError:
            result = []
        return result
