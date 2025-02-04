from openai import OpenAI
import asyncio
import trafilatura
import os, dotenv
from loguru import logger
from shared.settings import shared_settings
from prompting.base.duckduckgo_patch import PatchedDDGS
# Import the patched DDGS and use that

import numpy as np
from typing import Dict, List, Tuple

async def fetch_url(url: str) -> str:
    return trafilatura.fetch_url(url)

async def extract_content(content: str) -> str:
    return trafilatura.extract(content)


def create_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks of approximately chunk_size characters."""
    chunks = []
    current_chunk = ""
    
    for sentence in text.split('. '):
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def get_websites_with_similarity(
    query: str = "What are the 5 best phones I can buy this year?",
    n_results: int = 5,
    k: int = 3
) -> List[Dict[str, str]]:
    """
    Search for websites and return top K results based on embedding similarity.
    
    Args:
        query: Search query string
        n_results: Number of initial results to process
        k: Number of top similar results to return
        
    Returns:
        List of dictionaries containing website URLs and their best matching chunks
    """
    ddgs = PatchedDDGS()
    results = list(ddgs.text(query))
    urls = [r["href"] for r in results][:n_results]
    
    # Fetch and extract content
    content = await asyncio.gather(*[fetch_url(url) for url in urls])
    extracted = await asyncio.gather(*[extract_content(c) for c in content])
    
    # Create embeddings
    client = OpenAI(api_key=shared_settings.OPENAI_API_KEY)
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    ).data[0].embedding
    # Process each website
    results_with_similarity = []
    for url, html, text in zip(urls, content, extracted):
        if not text:  # Skip if extraction failed
            continue
            
        chunks = create_chunks(text)
        chunk_embeddings = client.embeddings.create(
            model="text-embedding-ada-002",
            input=chunks
        ).data
        
        # Find chunk with highest similarity
        similarities = [
            np.dot(query_embedding, chunk.embedding)
            for chunk in chunk_embeddings
        ]
        best_chunk_idx = np.argmax(similarities)
        
        results_with_similarity.append({
            "website": url,
            "best_chunk": chunks[best_chunk_idx],
            "similarity_score": similarities[best_chunk_idx],
            # "html": html,
            "text": text
        })
    
    # Sort by similarity score and return top K
    top_k = sorted(
        results_with_similarity,
        key=lambda x: x["similarity_score"],
        reverse=True
    )[:k]
    
    return [{
        "url": result["website"],
        "content": result["best_chunk"],
        # "html": result["html"],
        "relevant": result["text"]
    } for result in top_k]


# await get_websites_with_similarity(
#     "What are the 5 best phones I can buy this year?",
#     n_results=5, # number of initial websites to get
#     k=3 # number of top similar results to return
# )