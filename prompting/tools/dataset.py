# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import random
from typing import Dict
import requests
import datetime
import mathgenerator
import bittensor as bt
from datasets import load_dataset
from bs4 import BeautifulSoup
from collections.abc import Iterator
from sympy.parsing.latex import parse_latex

#TODO: Use beautiful soup to parse things like wikipedia articles and stack overflow questions and answers

class Dataset(Iterator):
    def __init__(self, dataset_id, seed=None):
        super().__init__()
        if seed is None:
            seed = random.randint(0, 1000)
        self.seed = seed
        self.dataset_id = dataset_id
        self.dataset = iter(
            load_dataset(dataset_id, split="train", streaming=True).shuffle(
                seed=seed, buffer_size=10000
            )
        )

    def __next__(self):
        while True:
            bt.logging.debug("Retrieving data from dataset...")
            text = next(self.dataset)["text"]

            # Check if the text is not empty or does not consist only of newline characters
            if text.strip():
                return {"text": text}
            
            
class MockDataset(Iterator):
    def __next__(self):
        return {"text": "What is the capital of Texas?"}


    
def chunk(text, sep, n_chunks=None):

    # choose a random chunk from the article
    chunks = [chunk for chunk in text.split(sep) if chunk.strip()]
    # select a subsequence of paragraphs
    if n_chunks is None:
        n_chunks = random.randint(1, len(chunks))
        
    start_chunk = random.randint(0, len(chunks) - n_chunks)
    bt.logging.info(f'Choosing {n_chunks} chunks starting at index {start_chunk}.')
    
    return sep.join(chunks[start_chunk:start_chunk + n_chunks])
            
            

class CodingDataset:
    
    all_languages = {
        "C++": [".cpp", ".hpp", ".c++", ".h++", ".cc", ".hh", ".C", ".H"],
        "CSS": [".css"],
        "Dockerfile": [".dockerfile", "Dockerfile"],
        "HTML":[".html"],
        "Java": [".java"],
        "JavaScript": [".js"],
        "Python": [".py"],
        "SQL": [".sql"],
        "Shell": [".sh", ".bash", ".command", ".zsh"],
    }
    
    def __init__(self, dataset_id='codeparrot/github-code', seed=None, languages=None):
        if seed is None:
            seed = random.randint(0, 1000)
        self.seed = seed
        
        if languages is None:
            languages = list(self.all_languages.keys())
        self.languages = languages    
        
        self.dataset_id = dataset_id
        self.dataset = iter(
            load_dataset(dataset_id, split="train", streaming=True, languages=self.languages).shuffle(
                seed=seed, buffer_size=10000
            )
        )

    def next(self, min_lines=5, max_lines=100):
        bt.logging.debug("Retrieving code from dataset...")
        while True:
            code = next(self.dataset)
            if min_lines <= len(code['code'].splitlines()) <= max_lines:
                return code




class WikiDataset:

    def __init__(self, min_length_words: int = 250, min_length_bytes: int = 1000, max_tries: int = 10, min_backlinks: int = 1):
        # Wikipedia API endpoint for a random article
        self.url = "https://en.wikipedia.org/w/api.php"
        self.min_length_words = min_length_words
        self.min_length_bytes = min_length_bytes
        self.max_tries = max_tries
        self.min_backlinks = min_backlinks

    def get_random_wikipedia_article(self) -> Dict:
        """sample random wikipedia article

        Args:
            min_length (int, optional): min number of words in article. Defaults to 1000.
            min_backlinks (int, optional): backlink is a hyperlink from one webpage to another webpage. Defaults to 1.
        """

        # Parameters for the API request
        params = {
            "action": "query",
            "format": "json",
            "prop": "info|linkshere|categories|categoryinfo|extracts",
            "generator": "random",
            "grnnamespace": 0,  # Namespace 0 indicates articles
            "grnlimit": 10,  # Number of random articles to fetch
            "inprop": "url|displaytitle|length",  # Requesting URL, title, and length of the page
            "lhprop": "pageid",  # Properties for links here (backlinks)
            "lhlimit": "max",  # Maximum number of backlinks to retrieve
            "exlimit": "max",  # Get extracts for each page
            "cllimit": "max",  # Get all categories for each page
        }

        tries = 0
        while tries < self.max_tries:
            # TODO: to avoid blocking from Wikipedia, we should provide a headers argument, where headers = {'User-Agent': 'Bittensor/0.0 (https://Bittensor.org; someone@opentensor.dev)'}
            response = requests.get(self.url, params=params)
            tries += 1

            data = response.json()
            if not data.get("query"):
                continue

            for page_id, page_info in data["query"]["pages"].items():
                length = page_info.get("length", 0)
                backlinks = len(page_info.get("linkshere", []))
                categories = [
                    cat.get("title", "").strip("Category:")
                    for cat in page_info.get("categories", [{}])
                ]
                # filter out any mention of articles
                categories = [cat for cat in categories if "article" not in cat.lower()]
                extract = page_info.get("extract")

                if (
                    length >= self.min_length_bytes and backlinks >= self.min_backlinks and extract
                ):  # and views >= min_views:
                    return {
                        "title": page_info["title"],
                        "url": page_info["fullurl"],
                        "length": length,
                        "extract": extract,
                        "backlinks": backlinks,
                        "categories": categories,
                    }

        raise Exception(
            f"Could not find an article with length >= {self.min_length_bytes} and backlinks >= {self.min_backlinks} after {self.max_tries} tries."
        )

    def get_wikipedia_article_content(self, title: str) -> str:
        """Return wikipedia article content

        Args:
            title (str): title of the article
            remove_headers (bool, optional): remove the headers in the content body. Defaults to False.

        Returns:
            str: article content
        """
        # Parameters for the API request to get article content
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,  # Get plain text content
        }

        # Making the API request
        # TODO: to avoid blocking from Wikipedia, we should provide a headers argument, where headers = {'User-Agent': 'Bittensor/0.0 (https://Bittensor.org; someone@opentensor.dev)'}
        response = requests.get(self.url, params=params)
        data = response.json()

        # Extracting the page content
        page = next(iter(data["query"]["pages"].values()))
        content = page.get("extract", "Content not found.")

        text = {None: ''}
        section = None
        for line in content.split("\n"):
            if line.startswith("==") and line.endswith("=="):
                section = line.strip("=").strip()
                text[section] = ""
                continue
            text[section] += line + "\n"

        return text

    def next(self, subset=False, chunk_sep='\n', n_chunks=None):
        bt.logging.debug("Retrieving data from dataset...")
        tries = 0
        while tries < self.max_tries:
            info = self.get_random_wikipedia_article()
            info['sections'] = self.get_wikipedia_article_content(info["title"])
            text = '\n'.join(info['sections'].values())
            tries += 1
            if len(text.split()) >= self.min_length_words:
                break
        
        if tries == self.max_tries:
            raise Exception(
                f"Could not find an article with length >= {self.min_length_words} words after {self.max_tries} tries."
            )
        
        if subset in info['sections'].keys():
            text = info['sections'][subset]
        elif subset:
            text = chunk(text, sep=chunk_sep, n_chunks=n_chunks)
        
        info['text'] = text
        return info



class StackOverflowDataset(Iterator):
    def __init__(self):
        # Stack Overflow API endpoint for a random article
        self.url = "https://api.stackexchange.com/2.3/questions"
        self.questions = []
        super().__init__()

    # @retry(
    #    stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=10)
    # )
    def get_stack_questions(self):
        url = "https://api.stackexchange.com/2.3/questions"
        params = {
            "order": "desc",
            "sort": "votes",  # Sorting by votes means that it's likely that the same questions will be fetched again
            "site": "stackoverflow",
            "pagesize": 100,  # Fetch 100 questions per API call
            "page": random.randint(1, 5),
        }

        # Fetch questions
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Parse response
        questions = response.json()["items"]

        # Filter questions by minimum upvotes
        min_upvotes = 10
        filtered_questions = [q for q in questions if q["score"] >= min_upvotes]
        # Shuffle the questions
        random.shuffle(filtered_questions)
        
        # Add the questions to the list of questions
        self.questions.extend(filtered_questions)
        return

    def get_stack_question(self) -> dict:
        # If the list of questions is empty, fetch more questions
        if not self.questions:
            self.get_stack_questions()
        question = self.questions.pop()
        # Fetch the highest voted answer for the selected question
        answer = self.get_stack_answer(question)
        return {'question': question["title"], 'answer':answer}

    # @retry(
    #    stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=10)
    # )
    def get_stack_answer(self, question):
        question_id = question["question_id"]
        url_answers = (
            f"https://api.stackexchange.com/2.3/questions/{question_id}/answers"
        )
        params_answers = {
            "order": "desc",
            "sort": "votes",
            "site": "stackoverflow",
            "filter": "withbody",  #'!9_bDDxJY5'
        }
        response_answers = requests.get(url_answers, params=params_answers)
        response_answers.raise_for_status()
        answers = response_answers.json()["items"]
        if not answers:
            bt.logging.warning("No answers found for the question!")

        highest_voted_answer = answers[0]  # The first answer is the highest voted
        soup = BeautifulSoup(highest_voted_answer["body"], "html.parser")
        full_content = soup.get_text(separator="\n")
        return full_content
    
    def __next__(self):
        while True:
            bt.logging.debug("Retrieving data from dataset...")
            info = self.get_stack_question()
            return info
        

class DateQADataset:
    def __init__(self, max_tries: int = 10):
        self.max_tries = max_tries

    def get_random_event(self) -> Dict:
        tries = 0
        while tries < self.max_tries:
            # TODO: to avoid blocking from Wikipedia, we should provide a headers argument, where headers = {'User-Agent': 'Bittensor/0.0 (https://Bittensor.org; someone@opentensor.dev)'}
            tries += 1

            # Step 1: Generate a random date
            year = 2000
            month = random.randint(1, 12)
            day = random.randint(1, 28)  # Simplified to avoid dealing with different month lengths
            random_date = datetime.date(year, month, day)

            # Step 2: Format the date for Wikipedia URL
            formatted_date = random_date.strftime("%B_%d")  # E.g., "January_01"

            # Step 3: Scrape Wikipedia
            url = f"https://en.wikipedia.org/wiki/{formatted_date}"
            print(f"Retrieving {url}...")
            response = requests.get(url)
            events = []

            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                available_sections = []
                for name in ['Events', 'Births', 'Deaths']:
                    section = soup.find('span', id=name)
                    if section:
                        available_sections.append(name)
                section = random.choice(available_sections)
                # Find the events section
                events_list = soup.find('span', id=section).parent.find_next_sibling('ul')
                
                for li in events_list.find_all('li'):
                    events.append(li)

                # Step 4: Extract Event Information and Step 5: Select an Event
                if events:
                    selected_event = random.choice(events)
                    links = selected_event.find_all('a')
                    #link_titles = [link.get("title") for link in links]
                    if links:
                        link = random.choice(links)
                    
                    return {'date': random_date.strftime('%B %d'), 'event': selected_event.get_text(), 'link': link.get("title")}

    def next(self):
        bt.logging.debug("Retrieving data from dataset...")
        info = self.get_random_event()
        return info

class MathDataset:

    def random_problem(self, parse):
        if parse:
            parseable_list = [2, 7, 11, 15, 19, 21, 24, 27, 29, 30, 32, 33, 35, 36, 42, 45, 48, 49, 52, 59, 60, 64, 66, 67, 68, 69, 70, 73, 76, 78, 81, 82, 83, 84, 85, 86, 87, 92, 94, 95, 96, 97, 105, 108, 109, 111, 115, 122, 123]
            options = parseable_list
            choice = random.choice((options))
            problem, solution = mathgenerator.genById(choice)
            #problem = parse_latex(str(problem).replace('$', '').strip())
            solution = parse_latex(str(solution).replace('$', '').strip()).evalf()
            return {'problem': problem, 'solution': solution}
        else:
            options = mathgenerator.getGenList()
            choice = random.choice(range(len(options)))
            problem, solution = mathgenerator.genById(choice)
            return {'problem': problem, 'solution': solution}
    
    def next(self, parse=True):
        return self.random_problem(parse)