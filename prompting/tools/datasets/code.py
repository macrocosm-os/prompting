# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
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

import re
import time
import random
import requests

import bittensor as bt
from bs4 import BeautifulSoup

from .base import Dataset
from ..selector import Selector
from datasets import load_dataset

languages = {
    "C++": {
        'keywords': ['auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'],
        'libraries': ['iostream', 'fstream', 'string', 'vector', 'map', 'set', 'algorithm', 'cmath', 'cstdio', 'cstdlib', 'ctime', 'cstring', 'cassert', 'cctype', 'cerrno', 'cfloat', 'ciso646', 'climits', 'clocale', 'cmath', 'csetjmp', 'csignal', 'cstdarg', 'cstddef', 'cstdio', 'cstdlib', 'cstring', 'ctime', 'cwchar', 'cwctype', 'complex', 'deque', 'exception', 'fstream', 'functional', 'iomanip', 'ios', 'iosfwd', 'iostream', 'istream', 'iterator', 'limits', 'list', 'locale', 'map', 'memory', 'new', 'numeric', 'ostream', 'queue', 'set', 'sstream', 'stack', 'stdexcept', 'streambuf', 'string', 'typeinfo', 'utility', 'valarray', 'vector'],
        'comments': ['//', '/*', '*/'],
    },
    "Dockerfile": {
        'keywords': ['from', 'maintainer', 'run', 'cmd', 'expose', 'env', 'add', 'copy', 'entrypoint', 'volume', 'user', 'workdir', 'onbuild'],
        'libraries': [],
        'comments': ['#']
    },
    "HTML": {
        'keywords': ['div', 'span', 'input', 'ul', 'body', 'tag', 'html', 'head', 'title', 'meta', 'link', 'script', 'style', 'a', 'img', 'table', 'label'],
        'libraries': [],
        'comments': ['<!--', '-->']
    },
    "Java": {
        'keywords': ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while'],
        'libraries': ['java.awt', 'java.awt.event', 'java.io', 'java.lang', 'java.math', 'java.net', 'java.text', 'java.util', 'javax.swing'],
        'comments': ['//', '/*', '*/', '*'],
    },
    "JavaScript": {
        'keywords': ['abstract', 'arguments', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'double', 'else', 'enum', 'eval', 'export', 'extends', 'false', 'final', 'finally', 'float', 'for', 'function', 'goto', 'if', 'implements', 'import', 'in', 'instanceof', 'int', 'interface', 'let', 'long', 'native', 'module.exports' 'new', 'null', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'true', 'try', 'typeof', 'var', 'void', 'volatile', 'while', 'with', 'yield'],
        'libraries': ['react', 'express','mongoose', 'axios', 'redux', 'react-redux', 'react-router-dom', 'react-dom', 'react-scripts', 'material-ui'],
        'comments': ['//', '/*', '*/']
    },
    "Python": {'keywords': ['False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'],
               'libraries': ['numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn', 'tensorflow', 'keras', 'pytorch', 'django', 'flask', 'requests', 'bs4', 'selenium', 'pyautogui', 'pyperclip', 'pyinputplus', 'pillow'],
               'comments': ['#']
    },
    "SQL": {'keywords': ['add', 'all', 'alter', 'and', 'any', 'as', 'asc', 'backup', 'between', 'case', 'check', 'column', 'constraint', 'create', 'database', 'default', 'delete', 'desc', 'distinct', 'drop', 'exec', 'exists', 'foreign', 'from', 'full', 'group', 'having', 'in', 'index', 'inner', 'insert', 'into', 'is', 'join', 'key', 'left', 'like', 'limit', 'not', 'null', 'on', 'or', 'order', 'outer', 'primary', 'procedure', 'right', 'rownum', 'select', 'set', 'table', 'top', 'truncate', 'union', 'unique', 'update', 'values', 'view', 'where'],
            'comments': ['--', '/*', '*/']
    },
    "Shell": {'keywords': ['alias', 'bg', 'bind', 'break', 'builtin', 'caller', 'cd', 'command', 'compgen', 'complete', 'continue', 'declare', 'dirs', 'disown', 'echo', 'enable', 'eval', 'exec', 'exit', 'export', 'false', 'fc', 'fg', 'getopts', 'hash', 'help', 'history', 'jobs', 'kill', 'let', 'local', 'logout', 'popd', 'printf', 'pushd', 'pwd', 'read', 'readonly', 'return', 'set', 'shift', 'shopt', 'source', 'suspend', 'test', 'times', 'trap', 'true', 'type', 'typeset', 'ulimit', 'umask', 'unalias', 'unset', 'wait'],
              'comments': ['#']
    },
}

class HFCodingDataset(Dataset):

    def __init__(
        self,
        dataset_id="codeparrot/github-code",
        seed=None,
        languages=None,
        buffer_size=10000,
    ):
        if seed is None:
            seed = random.randint(0, 1000)
        self.seed = seed

        if languages is None:
            languages = list(languages.keys())
        self.languages = languages

        self.dataset_id = dataset_id
        self.dataset = iter(
            load_dataset(
                dataset_id,
                split="train",
                streaming=True,
                languages=self.languages,
            ).shuffle(seed=seed, buffer_size=buffer_size)
        )

    def next(self, min_lines=5, max_lines=100):
        bt.logging.debug("Retrieving code from prompting.dataset...")
        t0 = time.time()
        while True:
            code = next(self.dataset)
            if min_lines <= len(code["code"].splitlines()) <= max_lines:
                code["fetch_time"] = time.time() - t0
                return code

    def search(self, query, min_lines=5, max_lines=100):
        # TODO: Would be great to be able to get other files from the same repo
        ...

    def filter_comments(self, code, language):
        # TODO: multiline comments
        # filter out comments

        # for start_tag, end_tag in languages[language]['multiline-comments']:
        #     code = re.sub(rf'{start_tag}.*?{end_tag}', '', code, flags=re.DOTALL)

        lines = []
        for line in code.splitlines():
            # TODO: use regex
            if any(line.strip().startswith(symbol) for symbol in languages[language]['comments']):
                continue

            lines.append(line.lower())

        return '\n'.join(lines)

    def extract_keywords(self, code, language, field):
        matches = set()

        # check which keywords and libraries are present in the code
        for keyword in languages[language].get(field,[]):
            if re.search(r'\b' + keyword + r'\b', code):
                matches.add(keyword)

        return matches

    def get_special_contents(self, code, language, remove_comments=True):

        if remove_comments:
            code = self.filter_comments(code, language)

        present_libraries = self.extract_keywords(code, language, 'libraries')
        present_keywords = self.extract_keywords(code, language, 'keywords')

        # here we select a library or a keyword as the forward term
        if present_libraries and random.random() < 0.7:
            forward_term = random.choice(list(present_libraries))
        elif present_keywords:
            forward_term = random.choice(list(present_keywords))
        else:
            forward_term = None

        return present_keywords, present_libraries, forward_term



class StackOverflowDataset:
    def __init__(self):
        # Stack Overflow API endpoint for a random article
        self.url = "https://api.stackexchange.com/2.3/questions"
        self.questions = []

    def get_stack_questions(self, min_upvotes=10):
        params = {
            "order": "desc",
            "sort": "votes",  # Sorting by votes means that it's likely that the same questions will be fetched again
            "site": "stackoverflow",
            "pagesize": 100,  # Fetch 100 questions per API call
            "page": random.randint(1, 5),
        }

        # Fetch questions
        response = requests.get(self.url, params=params)
        response.raise_for_status()

        # Parse response
        questions = response.json()["items"]

        # Filter questions by minimum upvotes
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
        return {"question": question["title"], "answer": answer}

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

    def next(self):
        bt.logging.debug("Retrieving data from prompting.dataset...")
        t0 = time.time()
        info = self.get_stack_question()
        info["fetch_time"] = time.time() - t0
        return info

