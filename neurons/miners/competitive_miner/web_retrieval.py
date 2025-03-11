import asyncio
from typing import Dict, List
import re
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import torch
import trafilatura
from angle_emb import AnglE

from prompting.base.duckduckgo_patch import PatchedDDGS
from shared import settings

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

async def fetch_url(url: str) -> str:
    return trafilatura.fetch_url(url)


async def extract_content(content: str) -> str:
    return trafilatura.extract(content)


def preprocess_text(text: str) -> str:
    """Clean and preprocess the text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove special characters but keep sentence punctuation
    text = re.sub(r'[^\w\s.,!?;]', '', text)
    return text.strip()


def create_chunks(text: str, chunk_size: int = 500, overlap: int = 100, min_length: int = 250) -> List[str]:
    """
    Split text into overlapping chunks using a sliding window approach.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk
        overlap: Number of characters to overlap between chunks
        min_length: Minimum length for a chunk to be considered valid
    """
    # Split into sentences first
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence would exceed chunk_size
        if current_length + sentence_length > chunk_size:
            if current_chunk:  # If we have a valid chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= min_length:
                    chunks.append(chunk_text)
                
                # Keep last sentences that fit within overlap size
                overlap_length = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if it meets minimum length
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= min_length:
            chunks.append(chunk_text)
    
    return chunks


class EmbeddingModel:
    def __init__(self):
        self.model = AnglE.from_pretrained(
            "WhereIsAI/UAE-Large-V1", 
            pooling_strategy="cls", 
            device=settings.shared_settings.NEURON_DEVICE
        ).to(settings.shared_settings.NEURON_DEVICE)
        
    @torch.no_grad()
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        embeddings = self.model.encode(texts, to_numpy=True)
        return embeddings

    def create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for a single text."""
        return self.create_embeddings([text])[0]

async def get_websites_with_similarity(
    query: str = "What are the 5 best phones I can buy this year?", 
    n_results: int = 5, 
    k: int = 3
) -> List[Dict[str, str]]:
    """
    Search for websites and return top K results based on embedding similarity.
    """
    ddgs = PatchedDDGS(proxy=settings.shared_settings.PROXY_URL, verify=False)
    results = list(ddgs.text(query))
    urls = [r["href"] for r in results][:n_results]

    # Fetch and extract content
    content = await asyncio.gather(*[fetch_url(url) for url in urls])
    extracted = await asyncio.gather(*[extract_content(c) for c in content])

    # Initialize embedding model
    embedding_model = EmbeddingModel()
    
    # Create query embedding
    query_embedding = embedding_model.create_embedding(query)

    results_with_similarity = []
    for url, html, text in zip(urls, content, extracted):
        if not text:
            continue

        chunks = create_chunks(text)
        if not chunks:
            continue
            
        # Get embeddings for original chunks
        chunk_embeddings = embedding_model.create_embeddings(chunks)

        # Calculate similarities using numpy for efficiency
        similarities = np.dot(chunk_embeddings, query_embedding)
        
        best_chunk_idx = np.argmax(similarities)
        
        results_with_similarity.append({
            "website": url,
            "best_chunk": chunks[best_chunk_idx],
            "similarity_score": float(similarities[best_chunk_idx]),
            "text": text,
        })

    # Sort by similarity score and return top K
    top_k = sorted(
        results_with_similarity, 
        key=lambda x: x["similarity_score"], 
        reverse=True
    )[:k]

    return [{
        "url": result["website"],
        "content": result["text"],
        "relevant": result["best_chunk"],
    } for result in top_k]