"""Expected miner's response is a JSON object with the following keys: url, content, relevant for each website."""

import json
import os
from collections import defaultdict
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from scipy import spatial
from thefuzz import fuzz

from prompting.datasets.random_website import DDGDataset, DDGDatasetEntry
from prompting.rewards.relevance import RelevanceRewardModel
from prompting.rewards.reward import BatchRewardOutput
from prompting.tasks.base_task import BaseTextTask
from shared.dendrite import DendriteResponseEvent

MIN_RELEVANT_CHARS = 300
MIN_MATCH_THRESHOLD = 98

# Define file paths
PAST_WEBSITES_FILE = "past_websites.csv"
TOP_DOMAINS_FILE = "data/top100k_domains.csv"

# Define blacklisted terms
BLACKLISTED_TERMS = {
    "howtogeek",
    "docs.google.com",
    "?q=",
    "/search",
    "sheets.google.com",
    "drive.google.com",
    "pastebin",
    "paste",
    "gist",
    "github",
    "gitlab",
    "bitbucket",
    "hastebin",
    "ghostbin",
    "privatebin",
}

# Maximum number of past URLs to store per user
N_PAST_URLS = 200

# Load the past_websites dictionary and top domains
try:
    # Load top domains
    top_domains_df = pd.read_csv(TOP_DOMAINS_FILE)
    TOP_DOMAINS = set(top_domains_df["Domain"].str.lower().values)

    # Load past websites
    if os.path.exists(PAST_WEBSITES_FILE):
        past_websites_df = pd.read_csv(PAST_WEBSITES_FILE)
        past_websites = defaultdict(list)
        # Group by uid and take only the last N_PAST_URLS entries
        for uid, group in past_websites_df.groupby("uid"):
            past_websites[uid] = group["domain"].tolist()[-N_PAST_URLS:]
    else:
        logger.warning(f"Past websites file {PAST_WEBSITES_FILE} does not exist, creating new dictionary")
        past_websites = defaultdict(list)
except Exception as e:
    logger.exception(f"Failed to load domains data: {e}")
    TOP_DOMAINS = set()
    past_websites = defaultdict(list)


def _append_to_past_websites(uid: str, domain: str):
    """Helper function to append domain to past_websites while maintaining max size."""
    past_websites[uid].append(domain)
    if len(past_websites[uid]) > N_PAST_URLS:
        past_websites[uid] = past_websites[uid][-N_PAST_URLS:]


class WebsiteResult(BaseModel):
    url: str | None
    content: str | None
    relevant: str | None


class WebRetrievalRewardModel(RelevanceRewardModel):
    def _cosine_similarity(self, content1: str, content2: str) -> float:
        """Calculate the cosine similarity between sentence embeddings of the reference and completions."""
        reference_emb_flatten = self.embedding_model.encode(content1, to_numpy=True).flatten()
        response_emb_flatten = self.embedding_model.encode(content2, to_numpy=True).flatten()
        return 1.0 - float(spatial.distance.cosine(reference_emb_flatten, response_emb_flatten))

    def score_website_result(
        self, dataset_entry: DDGDatasetEntry, response_url: str, response_content: str, response_relevant: str, uid: str
    ) -> float:
        if not response_url or not response_content or not response_relevant:
            return 0

        # Extract domain from URL
        parsed_url = urlparse(response_url)

        if any(term in response_url for term in BLACKLISTED_TERMS):
            logger.debug(f"Domain {parsed_url.netloc} contains blacklisted term, scoring 0")
            return 0

        netloc = parsed_url.netloc.lower()

        # Remove www. prefix if present
        if netloc.startswith("www."):
            netloc = netloc[4:]

        # Penalise a completion where the relevant section is contained in the URL (e.g. miners)
        # trying to use a search box to enter exactly the relevant section they need
        discount_factor = 1 - fuzz.token_sort_ratio(response_url, response_relevant) / 100
        # Check if URL is IP-based or has port
        if not netloc or any(c.isdigit() for c in netloc.split(".")) or ":" in netloc:
            discount_factor = 0
            logger.debug(f"URL {response_url} appears to be IP-based or on specific port, setting discount factor to 0")
            return 0
        else:
            domain = netloc

            # If domain is in top 100k, don't apply penalty
            if domain in TOP_DOMAINS:
                # if the domain is in the top 100k, we allow 10 occurrences in the last 200 URLs before penalising
                discount_factor *= 1.0 / (max(0, domain_count - 10))
                logger.debug(f"Domain {domain} is in top 100k domains, not applying penalty")
            else:
                # Count how many times this domain has been used by this miner
                domain_count = np.sum(np.array([domain == d for d in past_websites[uid]])) + 1
                discount_factor *= 1.0 / domain_count
                if domain in past_websites[uid]:
                    logger.debug(
                        f"Already used domain {domain} for this UID, applying ( discount ) factor {discount_factor}"
                    )
            _append_to_past_websites(uid, domain)

            # Content scraped from the URL provided in the completion.
            reference_website_content = DDGDataset.extract_website_content(response_url)
            if not reference_website_content or len(reference_website_content) == 0:
                logger.debug(f"Failed to extract miner's content from website: {response_url}")
                return 0

            if fuzz.ratio(response_content, reference_website_content) < MIN_MATCH_THRESHOLD:
                logger.debug("Miner returned text that doesn't match the website, scoring 0")
                return 0

            if len(response_relevant) > len(response_content) or len(response_relevant) < MIN_RELEVANT_CHARS:
                logger.debug(
                    f"Relevant section is too short (<{MIN_RELEVANT_CHARS} chars) or longer than the whole website content "
                    f"{len(response_relevant)} > {len(response_content)}"
                )
                return 0

            if response_relevant not in response_content:
                return 0

            return self._cosine_similarity(content1=dataset_entry.query, content2=response_relevant) * discount_factor

    def score_miner_response(
        self, dataset_entry: DDGDatasetEntry, completion: str, task: BaseTextTask | None = None, uid: str | None = None
    ) -> float:
        scores = []
        miner_websites: list[WebsiteResult] = self._parse_response(completion)
        unique_websites = np.unique([website.url for website in miner_websites])
        if unique_websites.size != len(miner_websites) and unique_websites.size != task.target_results:
            # logger.warning("Miner returned multiple websites with the same URL")
            return 0

        for website in miner_websites:
            scores.append(self.score_website_result(dataset_entry, website.url, website.content, website.relevant, uid))

        if scores:
            weights = np.arange(len(scores), 0, -1)
            return float(np.average(scores, weights=weights))
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

        for completion, uid in zip(response_event.completions, response_event.uids):
            rewards.append(self.score_miner_response(dataset_entry, completion, task=task, uid=uid))
            timings.append(0)

        logger.debug(f"REWARDWEBRETRIEVAL: {rewards}")
        logger.debug(f"COMPLETIONS: {response_event.completions}")

        # Save the past_websites dictionary to CSV
        past_websites_data = []
        for uid, domains in past_websites.items():
            for domain in domains:
                past_websites_data.append({"uid": uid, "domain": domain})
        pd.DataFrame(past_websites_data).to_csv(PAST_WEBSITES_FILE, index=False)

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
