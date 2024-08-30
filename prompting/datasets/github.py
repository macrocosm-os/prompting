from pydantic import BaseModel, model_validator
from prompting.datasets.base import DatasetEntry, BaseDataset
import copy
import requests
from loguru import logger
import random


def load_words(file_path):
    """Random words are used to create a random search term for the github API so we can find
    repositories to use for the coding challenge. If we find a better way to find a random repository
    that'd be great - but for now that's the best I got"""
    with open(file_path, "r") as file:
        words = file.read().splitlines()
    return words


WORDS = load_words("prompting/datasets/english_words.txt")

ALLOWED_FILE_ENDINGS = {
    "python": [".py"],
    "js": [".js", ".jsx", ".ts", ".tsx"],
}
MIN_FILE_SIZE = 100
MAX_FILE_SIZE = 100_000
MIN_INPUT_LINES = 10
OUTPUT_LINES = 10
MAX_LINES = 500

BRANCHES = ["main", "master", "dev", "development", "staging", "production", "prod", "testing", "test", "staging"]


class GithubDatasetEntry(DatasetEntry):
    github_url: str
    file_path: str
    file_content: str


def get_repositories(language, sort="stars", order="desc"):
    # TODO: We may want to introduce some contraints, e.g. a minimum number of stars to ensure decent code quality or so
    for _ in range(10):
        try:
            search_term = "+".join(random.sample(WORDS, 1))
            url = f"https://api.github.com/search/repositories?q={search_term}+language:{language}&sort={sort}&order={order}"
            logger.info(f"Searching for repositories with term: {search_term}")
            response = requests.get(url)
            response.raise_for_status()  # Check for request errors
            if response.json()["items"]:
                return response.json()["items"]
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
                    logger.error(f"Error fetching repositories: {ex}")

    def find_files(self):
        for branch in BRANCHES:
            files = requests.get(
                f"https://api.github.com/repos/{self.owner}/{self.name}/git/trees/{branch}?recursive=1"
            )
            if files.status_code == 200:
                self.branch = branch
                logger.info(f"Working with repository https://github.com/{self.owner}/{self.name} on branch {branch}")
                # print(f"branch \"{branch}\" found!")
                break
        file_names = [
            f["path"] for f in files.json()["tree"] if "size" in f.keys() and MIN_FILE_SIZE < f["size"] < MAX_FILE_SIZE
        ]
        self.valid_files = [f for f in file_names if f.endswith(tuple(ALLOWED_FILE_ENDINGS[self.language]))]
        return self.valid_files

    async def download_file(self, file: str):
        return requests.get(f"https://raw.githubusercontent.com/{self.owner}/{self.name}/{self.branch}/{file}").text

    async def _process_file(self, file_name: str) -> GithubDatasetEntry:
        file_content = await self.download_file(file_name)
        if len(file_content.split("\n")) < MIN_INPUT_LINES + OUTPUT_LINES:
            raise Exception(f"File {file_name} has too few lines")

        file_content = "\n".join(file_content.split("\n")[:MAX_LINES])
        logger.info(f"modifying file with {len(file_content.split("\n"))} lines")
        return GithubDatasetEntry(
            github_url=f"https://github.com/{self.owner}/{self.name}", file_path=file_name, file_content=file_content
        )

    async def next(self) -> GithubDatasetEntry:
        for i in range(len(self.valid_files) - self._current_file_idx):
            file_name = self.valid_files[self._current_file_idx]
            try:
                return await self._process_file(file_name)
            except Exception as ex:
                logger.error(f"Error processing file {file_name}: {ex}")
            self._current_file_idx += 1

    async def random(self) -> GithubDatasetEntry:
        valid_files = copy.deepcopy(self.valid_files)
        random.shuffle(valid_files)

        for file_name in valid_files:
            try:
                return await self._process_file(file_name)
            except Exception as ex:
                logger.error(f"Error processing file {file_name}: {ex}")


class GithubDataset(BaseDataset):
    def __init__(self, language: str = "python"):
        self.language = language
        self.current_repo = GithubRepo(language=language)

    async def get(self) -> GithubDatasetEntry:
        return await self.next()

    async def next(self) -> GithubDatasetEntry:
        return await self.current_repo.next()

    async def random(self) -> GithubDatasetEntry:
        self.reset()
        return await self.current_repo.random()

    async def reset(self):
        self.current_repo = GithubRepo(language=self.language)
