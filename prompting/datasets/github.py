from pydantic import BaseModel, model_validator
from prompting.datasets.base import DatasetEntry, BaseDataset
import copy
import requests
from loguru import logger
import random
from prompting.datasets.utils import ENGLISH_WORDS
import time

ALLOWED_FILE_ENDINGS = {
    "python": [".py"],
    "js": [".js", ".jsx", ".ts", ".tsx"],
}
MIN_FILE_SIZE = 100
MAX_FILE_SIZE = 100_000
MIN_INPUT_LINES = 10
OUTPUT_LINES = 10
MAX_LINES = 500
RETRIES = 5
WORD_FREQUENCY_THRESHOLD = 5000
BRANCHES = ["main", "master", "dev", "development", "staging", "production", "prod", "testing", "test", "staging"]


class GithubDatasetEntry(DatasetEntry):
    github_url: str
    file_path: str
    file_content: str


def get_repositories(language, sort="stars", order="desc"):
    # TODO: We may want to introduce some constraints, e.g. a minimum number of stars to ensure decent code quality or so
    for _ in range(10):
        try:
            search_term = "+".join(random.sample(ENGLISH_WORDS[:WORD_FREQUENCY_THRESHOLD], 1))
            url = f"https://api.github.com/search/repositories?q={search_term}+language:{language}&sort={sort}&order={order}"
            logger.info(f"Searching for repositories with term: {search_term}")
            response = requests.get(url)
            response.raise_for_status()  # Check for request errors
            if response.json()["items"]:
                if len(response.json()["items"]) == 0:
                    logger.warning(f"No repositories found for term: {search_term}")
                return response.json()["items"]
        except requests.exceptions.HTTPError as ex:
            if (response.status_code == 403 and "rate limit exceeded" in str(ex)) or (
                response.status_code == 429 and "too many requests" in str(ex)
            ):
                logger.warning("Rate limit exceeded. Waiting for 60 seconds before retrying...")
                time.sleep(60)
            else:
                logger.exception(ex)
                logger.error(f"Error fetching repositories: {ex}")
        except Exception as ex:
            logger.error(f"Error fetching repositories: {ex}")


class GithubRepo(BaseModel):
    language: str = "python"
    owner: str | None = None
    name: str | None = None
    branch: str | None = None
    valid_files: list[str] | None = None
    _current_file_idx: int = 0

    @model_validator(mode="after")
    def get_files(self) -> "GithubRepo":
        if not self.owner or not self.name:
            for _ in range(5):
                try:
                    repos = get_repositories(self.language)
                    repo = random.choice(repos)
                    self.owner, self.name = repo["owner"]["login"], repo["name"]
                    if self.find_files():
                        return self
                except Exception as ex:
                    logger.exception(ex)
                    logger.warning(f"Error fetching repositories: {ex}")
            logger.error("Failed to get a valid repository")

    def find_files(self):
        for branch in BRANCHES:
            files = requests.get(
                f"https://api.github.com/repos/{self.owner}/{self.name}/git/trees/{branch}?recursive=1"
            )
            if files.status_code == 200:
                self.branch = branch
                logger.info(f"Working with repository https://github.com/{self.owner}/{self.name} on branch {branch}")
                break
        jsons = files.json()
        logger.debug(jsons)
        file_names = [
            f["path"] for f in files.json()["tree"] if "size" in f.keys() and MIN_FILE_SIZE < f["size"] < MAX_FILE_SIZE
        ]
        self.valid_files = [f for f in file_names if f.endswith(tuple(ALLOWED_FILE_ENDINGS[self.language]))]
        return self.valid_files

    def download_file(self, file: str):
        return requests.get(f"https://raw.githubusercontent.com/{self.owner}/{self.name}/{self.branch}/{file}").text

    def _process_file(self, file_name: str) -> GithubDatasetEntry:
        file_content = self.download_file(file_name)
        if len(file_content.split("\n")) < (MIN_INPUT_LINES + OUTPUT_LINES):
            raise Exception(f"File {file_name} has too few lines")

        file_content = "\n".join(file_content.split("\n")[:MAX_LINES])
        n_lines = len(file_content.split("\n"))
        logger.info(f"modifying file with {n_lines} lines")
        return GithubDatasetEntry(
            github_url=f"https://github.com/{self.owner}/{self.name}", file_path=file_name, file_content=file_content
        )

    def next(self) -> GithubDatasetEntry:
        for i in range(len(self.valid_files) - self._current_file_idx):
            file_name = self.valid_files[self._current_file_idx]
            try:
                return self._process_file(file_name)
            except Exception as ex:
                logger.error(f"Error processing file {file_name}: {ex}")
            self._current_file_idx += 1

    def random(self) -> GithubDatasetEntry:
        valid_files = copy.deepcopy(self.valid_files)
        random.shuffle(valid_files)

        for file_name in valid_files:
            try:
                return self._process_file(file_name)
            except Exception as ex:
                logger.error(f"Error processing file {file_name}: {ex}")


class GithubDataset(BaseDataset):
    language: str = "python"
    current_repo: GithubRepo = None

    @model_validator(mode="after")
    def get_repo(self) -> "GithubDataset":
        self.current_repo = GithubRepo(language=self.language)
        return self

    def get(self) -> GithubDatasetEntry:
        return self.next()

    def next(self) -> GithubDatasetEntry:
        for _ in range(RETRIES):
            try:
                return self.current_repo.next()
            except Exception as ex:
                logger.error(f"Error getting next file: {ex}")
                self.reset()
        raise Exception("Failed to get a valid file")

    def random(self) -> GithubDatasetEntry:
        self.reset()
        return self.current_repo.random()

    def reset(self):
        self.current_repo = GithubRepo(language=self.language)
