import time
import aiohttp
import asyncio
import random
import bittensor as bt
from prompting.tools.datasets import Context
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


async def fetch_content(session: aiohttp.ClientSession, pageid: str) -> str:
    """
    Asynchronously fetches the plain text content of a Wikipedia page given its page ID.

    Args:
        session (aiohttp.ClientSession): The session used to make HTTP requests.
        pageid (str): The Wikipedia page ID of the page from which to fetch content.

    Returns:
        str: The plain text content of the Wikipedia page.

    Raises:
        aiohttp.ClientError: If there's an HTTP related error during the request.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "",
        "pageids": pageid,
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        content = data["query"]["pages"][str(pageid)]["extract"]
        return content


async def fetch_random_page(session: aiohttp.ClientSession) -> str:
    """
    Asynchronously fetches the page ID of a random Wikipedia page.

    Args:
        session (aiohttp.ClientSession): The session used to make HTTP requests.

    Returns:
        str: The page ID of a randomly selected Wikipedia page.

    Raises:
        aiohttp.ClientError: If there's an HTTP related error during the request.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": "0",
        "rnlimit": "1",
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        return data["query"]["random"][0]["id"]


async def fetch_page_details(
    session: aiohttp.ClientSession, pageid: str
) -> Dict[str, any]:
    """
    Asynchronously fetches detailed information about a Wikipedia page, including sections, links, categories, and external links.

    Args:
        session (aiohttp.ClientSession): The session used to make HTTP requests.
        pageid (str): The Wikipedia page ID from which to fetch details.

    Returns:
        Dict[str, Any]: A dictionary containing detailed information about the Wikipedia page.

    Raises:
        aiohttp.ClientError: If there's an HTTP related error during the request.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "format": "json",
        "pageid": pageid,
        "prop": "sections|links|categories|externallinks",
        "disabletoc": "1",
        "disableeditsection": "1",
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        return data["parse"]


async def fetch_random_section_context(
    session: aiohttp.ClientSession, progress: tqdm
) -> Context:
    """
    Asynchronously fetches the context of a random section from a random Wikipedia page, including title, topic, subtopic, content, and links.

    Args:
        session (aiohttp.ClientSession): The session used to make HTTP requests.
        progress (tqdm): A tqdm progress bar instance to update progress.

    Returns:
        Any: A context object containing various details about the section.

    Raises:
        SectionNotFoundException: If no valid section is found after filtering.
        MaxRetriesReachedException: If the maximum number of retry attempts is reached.
    """
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        try:
            request_time_start = time.time()
            bt.logging.info("Fetching random section context...")
            pageid = await fetch_random_page(session)
            page_details = await fetch_page_details(session, pageid)
            content = await fetch_content(session, pageid)

            # Filter sections here
            filtered_sections = [
                section
                for section in page_details["sections"]
                if section["line"] not in EXCLUDE_HEADERS
            ]

            if not filtered_sections:
                bt.logging.error("No valid sections found.")
                raise SectionNotFoundException("No valid sections found.")

            selected_section = random.choice(filtered_sections)

            internal_links = [
                link["*"] for link in page_details["links"] if link["ns"] == 0
            ]
            external_links = page_details.get("externallinks", [])
            tags = [
                category["*"]
                for category in page_details["categories"]
                if not any(excl in category["*"].lower() for excl in EXCLUDE_CATEGORIES)
            ]

            context = Context(
                title=page_details["title"],
                topic=selected_section.get("line", "No Topic"),
                subtopic=selected_section["line"],
                content=content,
                internal_links=internal_links,
                external_links=external_links,
                tags=tags,
                source="Wikipedia",
                extra={},
                stats={
                    "creator": fetch_random_section_context.__name__,
                    "fetch_time": time.time() - request_time_start,
                    "num_tries": attempt,
                },
            )
            progress.update(1)
            return context

        except SectionNotFoundException as e:
            bt.logging.warning(f"Attempt {attempt} failed: {e}")
            if attempt == max_attempts:
                bt.logging.error("Maximum retry attempts reached, failing...")
                raise MaxRetriesReachedException(
                    f"Maximum retry attempts reached: {max_attempts}"
                )


async def get_batch_random_sections(batch_size: int = 16) -> List[Context]:
    """
    Asynchronously fetches a batch of random sections from Wikipedia pages. This function utilizes concurrency to fetch multiple sections in parallel.

    Args:
        batch_size (int, optional): The number of random sections to fetch. Defaults to 16.

    Returns:
        List[Context]: A list of context objects, each containing details about a random section of a Wikipedia page.

    Details:
        The function creates an asynchronous session and a number of tasks equal to the batch size. Each task fetches a random section context from a Wikipedia page. All tasks are run concurrently, and the function waits for all tasks to complete before returning the results. A progress bar is displayed to track the progress of fetching the sections.

    Raises:
        aiohttp.ClientError: If there's an HTTP related error during any request in the tasks.
        SectionNotFoundException: If no valid section is found after filtering in any task.
        MaxRetriesReachedException: If the maximum number of retry attempts is reached in any task.
    """
    async with aiohttp.ClientSession() as session:
        tasks: List[asyncio.Task] = []
        progress = tqdm(
            total=batch_size,
            desc=f"Fetching {batch_size} random wikipedia sections",
            unit="section",
        )  # Total is the number of tasks

        # Creates a list of tasks to be executed concurrently
        for _ in range(batch_size):
            task = asyncio.create_task(fetch_random_section_context(session, progress))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        progress.close()  # Ensure the progress bar closes after all tasks complete

    return results
