import asyncio
import bittensor as bt
import sys
import random
from wikipedia import DisambiguationError, PageError
from wikipedia.exceptions import HTTPTimeoutError, WikipediaException, PageError
from functools import lru_cache
from typing import List
from prompting.utils.async_wiki_wrapper import AsyncWikipediaPage, _async_wiki_request
from dataclasses import dataclass

@dataclass
class ProcessedSection:
    header: str
    section_title: str
    content: List[str]
    
    def get_str_content(self):
        return "\n".join(self.content)

@dataclass
class ProcessedPage:
    title: str
    url: str
    page: AsyncWikipediaPage
    sections: List[ProcessedSection]
                        

async def process_page(
    page: AsyncWikipediaPage, valid_header: callable = None, valid_content: callable = None
) -> ProcessedPage:
    """Process a Wikipedia page and return a dictionary of sections with their content.

    Args:
        page: wikipedia.WikipediaPage
        valid_header: callable to determine if a section header is valid
        valid_content: callable to determine if a section content is valid
    Returns:
        dict: dictionary of sections and their content. Note that keys are tuples (header, section_title)
    """    
    header = ""
    page_sections = []

    # Get all section titles first
    section_titles = await page.sections 
    
    # Concurrently get the content of all sections
    contents = await asyncio.gather(
        *(page.section(section_title=title) for title in section_titles)
    )
        
    for section_title, content in zip(section_titles, contents):
        if not content:
            header = section_title
            continue

        # Filter out sections that don't match the headers and/or are not valid
        if (valid_header and not valid_header(header)) or (
            valid_content and not valid_content(content)
        ):
            continue

        section = ProcessedSection(header=header, section_title=section_title, content=content.splitlines())
        page_sections.append(section)

    if not page_sections:
        bt.logging.debug(f"No valid sections found in page {page.title!r} ({page.url})")
    
    return ProcessedPage(title=page.title, sections=page_sections, url=page.url, page=page)

async def process_pages(
    pages: List[AsyncWikipediaPage], valid_header: callable = None, valid_content: callable = None
):    
    tasks = [process_page(page, valid_header, valid_content) for page in pages]
    sections = await asyncio.gather(*tasks)
    return sections


async def search(query, results=10, suggestion=False):
    ''' Overwrites wikipedia base functions to use aiohttp and make it async '''
    search_params = {
        'list': 'search',
        'srprop': '',
        'srlimit': results,
        'limit': results,
        'srsearch': query
    }
    if suggestion:
        search_params['srinfo'] = 'suggestion'

    raw_results = await _async_wiki_request(search_params)

    if 'error' in raw_results:
        if raw_results['error']['info'] in ('HTTP request timed out.', 'Pool queue is full'):
            raise HTTPTimeoutError(query)
        else:
            raise WikipediaException(raw_results['error']['info'])

    search_results = (d['title'] for d in raw_results['query']['search'])

    if suggestion:
        if raw_results['query'].get('searchinfo'):
            return list(search_results), raw_results['query']['searchinfo']['suggestion']
        else:
            return list(search_results), None

    return list(search_results)

@lru_cache(maxsize=1000)
async def get_async_page(title=None, pageid=None, auto_suggest=True, redirect=True, preload=False, seed=None):
    try:
        if title is not None:
            if auto_suggest:
                results, suggestion = await search(title, results=1, suggestion=True)
                try:
                    title = suggestion or results[0]
                except IndexError:
                    raise PageError(title)
            # Assuming WikipediaPage is a class that needs to be defined or imported
            wiki_page = AsyncWikipediaPage(title=title, redirect=redirect, preload=preload)
            return wiki_page
        elif pageid is not None:
            wiki_page = AsyncWikipediaPage(pageid=pageid, preload=preload)
        else:
            raise ValueError("Either a title or a pageid must be specified")
    except DisambiguationError as e:
        bt.logging.debug(f"{e.__class__.__name__} loading page {title!r}: {e}")
        # exc info contains a tuple of (requested_title: str, possible_matches: List[str])
        pages = sys.exc_info()[1].args[1]
        if not type(pages) == list:
            return None
        title = random.Random(seed).choice(pages)
        return await get_async_page(title, auto_suggest=auto_suggest, redirect=redirect)

    except PageError as e:
        bt.logging.warning(f"{e.__class__.__name__} loading page {title!r}: {e}")
        if not auto_suggest:
            return await get_async_page(title, auto_suggest=True, redirect=redirect)
        return None


async def fetch_pages(titles, auto_suggest=False, redirect=True, preload=False):
    tasks = [get_async_page(title=title, auto_suggest=auto_suggest, redirect=redirect, preload=preload) for title in titles]
    pages = await asyncio.gather(*tasks)
    return pages
