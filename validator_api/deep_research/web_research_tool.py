from validator_api.deep_research.shared_resources import client
from loguru import logger
from typing import List
from duckduckgo_search import DDGS
import trafilatura
from validator_api.deep_research.shared_resources import SearchResult
# ================== Web Search Tool ==================

async def web_search_tool(query: str, num_results: int = 5) -> List[SearchResult]:
    """
    Perform a web search using DuckDuckGo and extract content using Trafilatura
    """
    logger.info(f"Starting web search for: {query}")
    
    try:
        async with DDGS() as ddgs:
            search_results = await ddgs.text(query, max_results=num_results)
            logger.debug(f"Found {len(search_results)} search results")
            
            results = []
            for result in search_results:
                url = result['href']
                try:
                    downloaded = await trafilatura.fetch_url(url)
                    if downloaded:
                        text = await trafilatura.extract(downloaded, include_links=True, include_images=True,
                                                         include_tables=True, no_fallback=False)
                        if text:
                            results.append(SearchResult(
                                url=url,
                                title=result['title'],
                                content=text
                            ))
                            logger.info(f"Successfully processed URL: {url}")
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
        
        return results
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

async def summarize_search_results(query: str, results: List[SearchResult]) -> str:
    """
    Summarize search results using OpenAI with more detailed analysis
    """
    logger.info("Starting search results summarization")
    
    if not results:
        logger.warning("No search results to summarize")
        return "No relevant information found from web search."
    
    context = "\n\n".join([
        f"Source: {result.url}\nTitle: {result.title}\n\n{result.content[:20000]}..."  # Increased content size
        for result in results
    ])
    
    prompt = f"""
You are a thorough research assistant that provides detailed summaries of web search results.

USER QUERY: {query}

WEB SEARCH RESULTS:
{context}

Please provide a comprehensive and detailed analysis of these search results that addresses the user query.
Your response should include:
1. A thorough summary of the main findings
2. Specific details and examples from the sources
3. Direct quotes where relevant (with source attribution)
4. Any conflicting information found across sources
5. Gaps in the available information
6. A confidence assessment of the information provided

Format your response with clear sections and bullet points where appropriate.
Always cite sources using [Source: URL] format.
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a detailed research assistant that provides thorough and well-structured summaries of web search results."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        summary = response.choices[0].message.content
        logger.info("Successfully generated detailed summary")
        logger.debug(f"Summary length: {len(summary)} characters")
        return summary
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return f"Error summarizing search results: {str(e)}"