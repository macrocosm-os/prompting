import aiohttp
import asyncio
import random
import bittensor as bt
from dataclasses import dataclass, field
from typing import List, Dict
from tqdm.asyncio import tqdm

EXCLUDE_HEADERS = ("See also", "References", "Further reading", "External links")
EXCLUDE_CATEGORIES = ("articles", "wiki", "pages", "cs1")

class SectionNotFoundException(Exception):
    """Exception raised when no valid section is found."""
    pass

class MaxRetriesReachedException(Exception):
    """Exception raised when maximum retry attempts are reached."""
    pass


@dataclass
class Context:
    title: str
    topic: str
    subtopic: str
    content: str
    internal_links: List[str]
    external_links: List[str]
    source: str
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, any] = field(default_factory=dict)
    stats: Dict[str, any] = field(default_factory=dict)

async def fetch_content(session: aiohttp.ClientSession, pageid: str) -> str:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "",
        "pageids": pageid
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        content = data['query']['pages'][str(pageid)]['extract']
        return content

async def fetch_random_page(session: aiohttp.ClientSession) -> str:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": "0",
        "rnlimit": "1"
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        return data['query']['random'][0]['id']

async def fetch_page_details(session: aiohttp.ClientSession, pageid: str) -> Dict[str, any]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "format": "json",
        "pageid": pageid,
        "prop": "sections|links|categories|externallinks",
        "disabletoc": "1",
        "disableeditsection": "1"
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        return data['parse']


async def fetch_random_section_context(session: aiohttp.ClientSession, progress: tqdm) -> Context:
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        try:
            bt.logging.info("Fetching random section context...")
            pageid = await fetch_random_page(session)
            page_details = await fetch_page_details(session, pageid)
            content = await fetch_content(session, pageid)

            # Filter sections here          
            filtered_sections = [section for section in page_details['sections'] if section['line'] not in EXCLUDE_HEADERS]

            if not filtered_sections:
                bt.logging.error("No valid sections found.")
                raise SectionNotFoundException("No valid sections found.")

            selected_section = random.choice(filtered_sections)
            
            internal_links = [link['*'] for link in page_details['links'] if link['ns'] == 0]
            external_links = page_details.get('externallinks', [])
            tags = [category['*'] for category in page_details['categories'] if not any(excl in category['*'].lower() for excl in EXCLUDE_CATEGORIES)]
            
            context = Context(
                title=page_details['title'],
                topic=selected_section.get('line', 'No Topic'),
                subtopic=selected_section['line'],
                content=content,
                internal_links=internal_links,
                external_links=external_links,
                tags=tags,
                source="Wikipedia",
                extra={}
            )
            progress.update(1)
            return context

        except SectionNotFoundException as e:
            bt.logging.warning(f"Attempt {attempt} failed: {e}")
            if attempt == max_attempts:
                bt.logging.error("Maximum retry attempts reached, failing...")
                raise MaxRetriesReachedException(f"Maximum retry attempts reached: {max_attempts}")
                                    

async def get_batch_random_sections(batch_size: int = 16) -> List[Context]:    
    async with aiohttp.ClientSession() as session:                
        tasks: List[asyncio.Task] = []
        progress = tqdm(total=batch_size, desc=f"Fetching {batch_size} random wikipedia sections", unit="section")  # Total is the number of tasks
        
        # Creates a list of tasks to be executed concurrently
        for _ in range(batch_size):
            task = asyncio.create_task(fetch_random_section_context(session, progress))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        progress.close()  # Ensure the progress bar closes after all tasks complete    
        
    return results