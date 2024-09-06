import re
import time

import numpy as np
import re

from prompting.base.dendrite import DendriteResponseEvent
from prompting.rewards.reward import BaseRewardModel, BatchRewardOutput
from prompting.rewards.web_retrieval import WebRetrievalRewardModel
from prompting.datasets.random_website import DDGDatasetEntry

import nltk
from nltk.corpus import words
LENGTH_THRESHOLD = 500


k = 3


async def get_text(url):
    website = trafilatura.fetch_url(url)
    return trafilatura.extract(website)


async def get_top_results(query, results, k=3):
    query_embedding = (await get_embeddings([query]))[0]
    result_embeddings = await get_embeddings([result["body"] for result in results])
    similarities = [query_embedding @ result_embedding for result_embedding in result_embeddings]

    top_results = np.array(results)[np.argsort(similarities)[-k:]]
    return top_results, query_embedding


async def process_sections(top_results: np.ndarray):
    texts = [get_text(result["href"]) for result in top_results]
    texts= await asyncio.gather(*texts)


    sections = []
    for text, result in zip(texts, top_results):
        sections.append({"url": result["href"], "text": ""})
        
        sentences = re.split(r"(?<=\.[ \n])", text)
        for sentence in sentences:
            if len(sections[-1]["text"]) > LENGTH_THRESHOLD:
                sections.append({"url": result["href"], "text": ""})
            sections[-1]["text"] += sentence
    return [s for s in sections if len(s["text"]) >= LENGTH_THRESHOLD]


async def get_top_sections(query_embedding, sections):
    section_texts = [s["text"] for s in sections]
    embeddings = await get_embeddings(section_texts)
    similarities = [query_embedding @ embedding for embedding in embeddings]
    for section, sim in zip(sections, similarities):
        section["score"] = sim
    return sorted(sections, key=lambda x: x["score"], reverse=True)


class WebRetrievalRewardModel(BaseRewardModel):
    MIN_MATCH_THRESHOLD = 98
    ACCURACY_THRESHOLD = 0.001

    async def _reward(result: MinerResult, dataset_entry: DDGDatasetEntry) -> float:
        query_embedding = (await get_embeddings([dataset_entry.query]))[0]
        result_embedding = (await get_embeddings([result.text]))[0]

        score = query_embedding @ result_embedding
        website_text = await get_text(result.url)

        if not (1 - ACCURACY_THRESHOLD < score / result.score < 1 + ACCURACY_THRESHOLD):
            logger.info(f"Miner reported incorrect result, scoring 0. Miner returned: {result.score}, actual score: {score}")
            return 0
        if fuzz.token_set_ratio(website_text, result.text) < MIN_MATCH_THRESHOLD:
            logger.info("Miner returned text that doesn't match the website, scoring 0")
            return 0
        return score

    @property
    def name(self) -> str:
        return "web_retrieval"

    def reward(self, reference: str, response_event: DendriteResponseEvent) -> BatchRewardOutput:
        """Compute difference scores given a completion and reference pair."""
        rewards = []
        timings = []
        t0 = time.perf_counter()
        completions: list[str] = response_event.completions

        rewards.append()

        output = BatchRewardOutput(
            rewards=np.asarray(rewards),
            timings=np.array([time.perf_counter() - t0]),
            # extra_info={
            #     "type": self.name,
            # },
        )
        return output
