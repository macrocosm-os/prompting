import aiohttp
import asyncio
import bittensor as bt
import sys
import random
from datetime import datetime, timedelta
from wikipedia import USER_AGENT, RATE_LIMIT, RATE_LIMIT_MIN_WAIT, API_URL, WikipediaPage, DisambiguationError, PageError
from wikipedia.exceptions import HTTPTimeoutError, WikipediaException, PageError
from functools import lru_cache
from typing import Dict, List



##################### Wraps and overwrite wikipedia features to make it async #####################

async def _async_wiki_request(params):
    global RATE_LIMIT_LAST_CALL
    global USER_AGENT

    params['format'] = 'json'
    if 'action' not in params:
        params['action'] = 'query'

    headers = {'User-Agent': USER_AGENT}

    if RATE_LIMIT and RATE_LIMIT_LAST_CALL and \
        RATE_LIMIT_LAST_CALL + RATE_LIMIT_MIN_WAIT > datetime.now():

        wait_time = (RATE_LIMIT_LAST_CALL + RATE_LIMIT_MIN_WAIT) - datetime.now()
        await asyncio.sleep(wait_time.total_seconds())

    async with aiohttp.ClientSession() as session:
        async with session.get(API_URL, params=params, headers=headers) as response:
            if RATE_LIMIT:
                RATE_LIMIT_LAST_CALL = datetime.now()
            return await response.json()
        
        
class AsyncWikipediaPage(WikipediaPage):
    @property
    async def sections(self):
        '''
        Overwrites the `sections` property to be async.
        '''
        if not getattr(self, '_sections', False):
            query_params = {
            'action': 'parse',
            'prop': 'sections',
            }
            if not getattr(self, 'title', None) is None:
                query_params["page"] = self.title

            request = await _async_wiki_request(query_params)
            self._sections = [section['line'] for section in request['parse']['sections']]

        return self._sections
        
    async def section(self, section_title: str):
        '''
        Get the plain text content of a section from `self.sections`.
        Returns None if `section_title` isn't found, otherwise returns a whitespace stripped string.

        This is a convenience method that wraps self.content.

        .. warning:: Calling `section` on a section that has subheadings will NOT return
            the full text of all of the subsections. It only gets the text between
            `section_title` and the next subheading, which is often empty.
        '''

        section = u"== {} ==".format(section_title)        
        content = await self.content
        
        try:
            index = content.index(section) + len(section)
        except ValueError:
            return None

        try:
            next_index = content.index("==", index)
        except ValueError:
            next_index = len(content)

        return content[index:next_index].lstrip("=").strip()
    
    @property
    async def content(self):
        '''
        Overwrites the `content` property that is called by the `section` property.
        This change enables the `content` property to be called independently in async.
        '''

        if not getattr(self, '_content', False):
            query_params = {
                'prop': 'extracts|revisions',
                'explaintext': '',
                'rvprop': 'ids'
            }
            if not getattr(self, 'title', None) is None:
                query_params['titles'] = self.title
            else:
                query_params['pageids'] = self.pageid
            request = await _async_wiki_request(query_params)
            self._content     = request['query']['pages'][self.pageid]['extract']
            self._revision_id = request['query']['pages'][self.pageid]['revisions'][0]['revid']
            self._parent_id   = request['query']['pages'][self.pageid]['revisions'][0]['parentid']

        return self._content
                

##################### Utility functions #####################

async def process_page(
    page: AsyncWikipediaPage, valid_header: callable = None, valid_content: callable = None
) -> Dict:
    """Process a Wikipedia page and return a dictionary of sections with their content.

    Args:
        page: wikipedia.WikipediaPage
        valid_header: callable to determine if a section header is valid
        valid_content: callable to determine if a section content is valid
    Returns:
        dict: dictionary of sections and their content. Note that keys are tuples (header, section_title)
    """
    header = ""
    sections = {}

    for section_title in await page.sections:
        content = await page.section(section_title=section_title)
        if not content:
            header = section_title
            continue

        # Filter out sections that don't match the headers and/or are not valid
        if (valid_header and not valid_header(header)) or (
            valid_content and not valid_content(content)
        ):
            continue

        key = (header, section_title)
        sections[key] = content.splitlines()

    if not sections:
        bt.logging.debug(f"No valid sections found in page {page.title!r} ({page.url})")

    return sections

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
            return AsyncWikipediaPage(title, redirect=redirect, preload=preload)
        elif pageid is not None:
            return AsyncWikipediaPage(pageid=pageid, preload=preload)
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
